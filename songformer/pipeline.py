"""SongFormerPipeline — installable wrapper around the SongFormer research code.

Loads MuQ + MusicFM SSL encoders and the SongFormer MSA head from the HuggingFace
cache, then exposes a single :py:meth:`analyze` entry point that returns labeled
song sections.

The heavy lifting (model architecture, post-processing, rule cleanup) is reused
from ``src/SongFormer/`` — this module wires it up behind a clean API.
"""

from __future__ import annotations

import importlib.resources
import math
import os
import sys
from pathlib import Path
from typing import Callable, TypeAlias

import numpy as np

ProgressFn: TypeAlias = Callable[[str, float], None]
"""Optional callback for `analyze()` progress.

Signature: ``progress(stage: str, fraction: float)`` where ``stage`` is one of
``load_audio | encode | infer | postprocess | done`` and ``fraction`` is in
``[0.0, 1.0]``. Always monotonically non-decreasing across a single call.
"""

# msaf transitively reads ``scipy.inf`` which was removed in modern scipy.
# Patch BEFORE importing the upstream model module that imports msaf at top level.
import scipy

if not hasattr(scipy, "inf"):
    scipy.inf = np.inf  # type: ignore[attr-defined]

import librosa
import torch
from ema_pytorch import EMA
from omegaconf import OmegaConf

from songformer.types import AnalysisResult, Segment

# Make the vendored MusicFM importable. Two entries are needed:
#   - ``songformer/vendor/``         → enables ``from musicfm.model.X import Y``
#   - ``songformer/vendor/musicfm/`` → enables ``from modules.X import Y`` that
#                                       musicfm_25hz.py does internally.
_VENDOR_ROOT = Path(__file__).parent / "vendor"
_MUSICFM_DIR = _VENDOR_ROOT / "musicfm"
for _p in (_VENDOR_ROOT, _MUSICFM_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Make the upstream src/SongFormer modules importable as ``from dataset.X``,
# ``from postprocessing.X``, ``from models.X`` — minimal-change requirement.
_UPSTREAM_DIR = Path(__file__).parent.parent / "src" / "SongFormer"
if _UPSTREAM_DIR.exists() and str(_UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_DIR))

from muq import MuQ  # noqa: E402
from musicfm.model.musicfm_25hz import MusicFM25Hz  # noqa: E402

from dataset.label2id import (  # noqa: E402
    DATASET_ID_ALLOWED_LABEL_IDS,
    DATASET_LABEL_TO_DATASET_ID,
)
from models.SongFormer import Model  # noqa: E402


_DATASET_LABEL = "SongForm-HX-8Class"
_INPUT_SAMPLING_RATE = 24000
_TIME_DUR = 420
_AFTER_DOWNSAMPLING_FRAME_RATES = 8.333
_WIN_SIZE = 420
_HOP_SIZE = 420
_NUM_CLASSES = 128

_MUSICFM_REPO = "minzwon/MusicFM"
_SONGFORMER_REPO = "ASLP-lab/SongFormer"
_SONGFORMER_WEIGHTS = "SongFormer.safetensors"
_MUSICFM_WEIGHTS = "pretrained_msd.pt"
_MUSICFM_STATS = "msd_stats.json"
_MUQ_REPO = "OpenMuQ/MuQ-large-msd-iter"


def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _hf_cached_or_raise(repo_id: str, filename: str) -> str:
    """Return a path to a HuggingFace-cached file without downloading.

    Raises a clear error pointing the caller at ``songformer-download`` if the
    file is not present.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            f"SongFormer weights not found in the HuggingFace cache "
            f"({repo_id}/{filename}). Run `songformer-download` first."
        ) from exc


def _load_config() -> OmegaConf:
    config_path = importlib.resources.files("songformer") / "configs" / "SongFormer.yaml"
    return OmegaConf.load(str(config_path))


class SongFormerPipeline:
    """End-to-end MSA pipeline: audio file → labeled segments."""

    def __init__(
        self,
        muq_model: MuQ,
        musicfm_model: MusicFM25Hz,
        msa_model: Model,
        config,
        device: str,
    ) -> None:
        self.muq = muq_model
        self.musicfm = musicfm_model
        self.msa = msa_model
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        flash_attention: bool | None = None,
    ) -> "SongFormerPipeline":
        device = device or _auto_device()
        if flash_attention is None:
            flash_attention = device.startswith("cuda")

        config = _load_config()
        OmegaConf.set_struct(config, False)
        config.attn_flash = bool(flash_attention)

        muq = MuQ.from_pretrained(_MUQ_REPO)
        muq = muq.to(device).eval()

        musicfm_stats = _hf_cached_or_raise(_MUSICFM_REPO, _MUSICFM_STATS)
        musicfm_weights = _hf_cached_or_raise(_MUSICFM_REPO, _MUSICFM_WEIGHTS)
        musicfm = MusicFM25Hz(
            is_flash=False,
            stat_path=musicfm_stats,
            model_path=musicfm_weights,
        )
        musicfm = musicfm.to(device).eval()

        msa_model = Model(config)
        songformer_weights = _hf_cached_or_raise(_SONGFORMER_REPO, _SONGFORMER_WEIGHTS)
        from safetensors.torch import load_file

        state = load_file(songformer_weights, device="cpu")
        model_ema = EMA(msa_model, include_online_model=False)
        model_ema.load_state_dict(state)
        msa_model.load_state_dict(model_ema.ema_model.state_dict())
        msa_model = msa_model.to(device).eval()

        return cls(
            muq_model=muq,
            musicfm_model=musicfm,
            msa_model=msa_model,
            config=config,
            device=device,
        )

    def analyze(
        self,
        audio_path: str | os.PathLike[str],
        progress: ProgressFn | None = None,
    ) -> AnalysisResult:
        from postprocessing.functional import postprocess_functional_structure

        def emit(stage: str, fraction: float) -> None:
            if progress is None:
                return
            try:
                progress(stage, max(0.0, min(1.0, fraction)))
            except Exception:
                # Never let a buggy callback take down inference.
                pass

        emit("load_audio", 0.0)
        audio_path = str(audio_path)
        wav, _sr = librosa.load(audio_path, sr=_INPUT_SAMPLING_RATE)
        duration = float(len(wav)) / _INPUT_SAMPLING_RATE
        audio = torch.tensor(wav).to(self.device)
        emit("load_audio", 0.05)

        win_size = _WIN_SIZE
        hop_size = _HOP_SIZE
        total_len = (
            (audio.shape[0] // _INPUT_SAMPLING_RATE) // _TIME_DUR
        ) * _TIME_DUR + _TIME_DUR
        total_frames = math.ceil(total_len * _AFTER_DOWNSAMPLING_FRAME_RATES)

        function_logits_acc = np.zeros([total_frames, _NUM_CLASSES])
        boundary_logits_acc = np.zeros([total_frames])
        function_counts = np.zeros([total_frames, _NUM_CLASSES])
        boundary_counts = np.zeros([total_frames])

        label_mask = np.ones(_NUM_CLASSES, dtype=bool)
        label_mask[DATASET_ID_ALLOWED_LABEL_IDS[DATASET_LABEL_TO_DATASET_ID[_DATASET_LABEL]]] = False
        label_id_masks = (
            torch.from_numpy(label_mask).to(self.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        )
        dataset_ids = torch.tensor(
            [DATASET_LABEL_TO_DATASET_ID[_DATASET_LABEL]],
            device=self.device,
            dtype=torch.long,
        )

        # Pre-compute a rough total step count so progress fractions are smooth.
        # Each outer iteration does: 2 full-window encodes (MuQ + MusicFM), up to
        # `hop_size / 30` inner 30s encodes, and one MSA infer. The estimate is
        # generous on the last outer iteration (real inner count may be lower for
        # short songs), so the fraction reported is conservative — we clamp at
        # 0.95 below to leave headroom for post-processing.
        n_outer = max(1, total_len // hop_size)
        inner_per_outer = hop_size // 30
        total_steps = max(1, n_outer * (2 + inner_per_outer + 1))
        step = 0
        main_lo, main_hi = 0.05, 0.95  # progress band reserved for the main loop

        lens = 0
        i = 0
        with torch.no_grad():
            while True:
                start_idx = i * _INPUT_SAMPLING_RATE
                end_idx = min((i + win_size) * _INPUT_SAMPLING_RATE, audio.shape[-1])
                if start_idx >= audio.shape[-1]:
                    break
                if end_idx - start_idx <= 1024:
                    break

                audio_seg = audio[start_idx:end_idx]

                muq_out = self.muq(audio_seg.unsqueeze(0), output_hidden_states=True)
                muq_embd_420s = muq_out["hidden_states"][10]
                del muq_out
                step += 1
                emit("encode", main_lo + (step / total_steps) * (main_hi - main_lo))

                _, musicfm_hidden = self.musicfm.get_predictions(audio_seg.unsqueeze(0))
                musicfm_embd_420s = musicfm_hidden[10]
                del musicfm_hidden
                step += 1
                emit("encode", main_lo + (step / total_steps) * (main_hi - main_lo))

                wraped_muq_30s = []
                wraped_musicfm_30s = []
                for idx_30s in range(i, i + hop_size, 30):
                    s30 = idx_30s * _INPUT_SAMPLING_RATE
                    e30 = min(
                        (idx_30s + 30) * _INPUT_SAMPLING_RATE,
                        audio.shape[-1],
                        (i + hop_size) * _INPUT_SAMPLING_RATE,
                    )
                    if s30 >= audio.shape[-1]:
                        break
                    if e30 - s30 <= 1024:
                        continue
                    wraped_muq_30s.append(
                        self.muq(audio[s30:e30].unsqueeze(0), output_hidden_states=True)[
                            "hidden_states"
                        ][10]
                    )
                    wraped_musicfm_30s.append(
                        self.musicfm.get_predictions(audio[s30:e30].unsqueeze(0))[1][10]
                    )
                    step += 1
                    emit("encode", main_lo + (step / total_steps) * (main_hi - main_lo))

                wraped_muq = torch.concatenate(wraped_muq_30s, dim=1)
                wraped_musicfm = torch.concatenate(wraped_musicfm_30s, dim=1)
                all_embds = [wraped_musicfm, wraped_muq, musicfm_embd_420s, muq_embd_420s]
                min_len = min(x.shape[1] for x in all_embds)
                all_embds = [x[:, :min_len, :] for x in all_embds]
                embd = torch.concatenate(all_embds, axis=-1)

                _msa_info, chunk_logits = self.msa.infer(
                    input_embeddings=embd,
                    dataset_ids=dataset_ids,
                    label_id_masks=label_id_masks,
                    with_logits=True,
                )
                step += 1
                emit("infer", main_lo + (step / total_steps) * (main_hi - main_lo))

                start_frame = int(i * _AFTER_DOWNSAMPLING_FRAME_RATES)
                end_frame = start_frame + min(
                    math.ceil(hop_size * _AFTER_DOWNSAMPLING_FRAME_RATES),
                    chunk_logits["boundary_logits"][0].shape[0],
                )

                function_logits_acc[start_frame:end_frame, :] += (
                    chunk_logits["function_logits"][0].detach().cpu().numpy()
                )
                boundary_logits_acc[start_frame:end_frame] = (
                    chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                )
                function_counts[start_frame:end_frame, :] += 1
                boundary_counts[start_frame:end_frame] += 1
                lens += end_frame - start_frame
                i += hop_size

        function_logits_acc /= np.maximum(function_counts, 1)
        boundary_logits_acc /= np.maximum(boundary_counts, 1)

        logits = {
            "function_logits": torch.from_numpy(function_logits_acc[:lens]).unsqueeze(0),
            "boundary_logits": torch.from_numpy(boundary_logits_acc[:lens]).unsqueeze(0),
        }

        emit("postprocess", 0.97)
        msa_output = postprocess_functional_structure(logits, self.config)
        assert msa_output[-1][-1] == "end"
        msa_output = _rule_post_processing(msa_output)

        segments: list[Segment] = []
        for idx in range(len(msa_output) - 1):
            segments.append(
                Segment(
                    start=float(msa_output[idx][0]),
                    end=float(msa_output[idx + 1][0]),
                    label=str(msa_output[idx][1]),
                )
            )

        emit("done", 1.0)
        return AnalysisResult(segments=segments, duration=duration)


def _rule_post_processing(msa_list):
    """Lifted verbatim from src/SongFormer/infer/infer.py."""
    if len(msa_list) <= 2:
        return msa_list
    result = msa_list.copy()
    while len(result) > 2:
        first_duration = result[1][0] - result[0][0]
        if first_duration < 1.0 and len(result) > 2:
            result[0] = (result[0][0], result[1][1])
            result = [result[0]] + result[2:]
        else:
            break
    while len(result) > 2:
        last_label_duration = result[-1][0] - result[-2][0]
        if last_label_duration < 1.0:
            result = result[:-2] + [result[-1]]
        else:
            break
    while len(result) > 2:
        if result[0][1] == result[1][1] and result[1][0] <= 10.0:
            result = [(result[0][0], result[0][1])] + result[2:]
        else:
            break
    while len(result) > 2:
        last_duration = result[-1][0] - result[-2][0]
        if result[-2][1] == result[-3][1] and last_duration <= 10.0:
            result = result[:-2] + [result[-1]]
        else:
            break
    return result
