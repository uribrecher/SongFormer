# SongFormer Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the SongFormer fork into an installable Python library with a clean `SongFormerPipeline` API.

**Architecture:** A `songformer/` package at the repo root wraps the existing `src/SongFormer/` code. MusicFM is vendored into `songformer/vendor/`. The pipeline class encapsulates model initialization, embedding extraction, inference, and post-processing. Weights are managed via `huggingface_hub`.

**Tech Stack:** Python 3.11+, PyTorch, huggingface_hub, muq, x-transformers, ema-pytorch, omegaconf, pydantic, librosa

**Spec:** See `docs/superpowers/specs/2026-04-25-songformer-library-design.md` (in the audio-analysis-mcp repo).

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | NEW — package metadata, dependencies, CLI entry point |
| `songformer/__init__.py` | NEW — public exports: `SongFormerPipeline`, `Segment`, `AnalysisResult` |
| `songformer/types.py` | NEW — Pydantic models: `Segment`, `AnalysisResult` |
| `songformer/download.py` | NEW — weight download CLI via `huggingface_hub` |
| `songformer/pipeline.py` | NEW — `SongFormerPipeline` class wrapping all inference logic |
| `songformer/vendor/musicfm/` | NEW — vendored copy of MusicFM package |
| `src/SongFormer/models/SongFormer.py` | MODIFY — make `attn_flash` configurable in `WrapedTransformerEncoder` and `Model` |

---

### Task 1: Add pyproject.toml

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "songformer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "torchaudio>=2.4",
    "librosa>=0.11",
    "muq>=0.1",
    "transformers>=4.51",
    "huggingface-hub",
    "x-transformers>=2.4",
    "ema-pytorch>=0.7",
    "omegaconf",
    "scipy",
    "numpy",
    "pydantic>=2.0",
    "safetensors",
    "einops",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]

[project.scripts]
songformer-download = "songformer.download:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Verify the package resolves**

Run: `uv sync --dev`
Expected: Dependencies install successfully. There will be import errors for `songformer` itself since the package doesn't exist yet — that's fine.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml with dependencies and CLI entry point"
```

---

### Task 2: Vendor MusicFM

**Files:**
- Create: `songformer/vendor/__init__.py`
- Create: `songformer/vendor/musicfm/__init__.py`
- Create: `songformer/vendor/musicfm/model/__init__.py`
- Create: `songformer/vendor/musicfm/model/musicfm_25hz.py`
- Create: `songformer/vendor/musicfm/modules/__init__.py`
- Create: `songformer/vendor/musicfm/modules/conv.py`
- Create: `songformer/vendor/musicfm/modules/features.py`
- Create: `songformer/vendor/musicfm/modules/random_quantizer.py`

- [ ] **Step 1: Initialize the git submodule to get MusicFM source**

```bash
git submodule update --init src/third_party/musicfm
```

Expected: The `src/third_party/musicfm/` directory is populated with `model/`, `modules/`, etc.

- [ ] **Step 2: Create the vendor directory structure**

```bash
mkdir -p songformer/vendor/musicfm/model
mkdir -p songformer/vendor/musicfm/modules
```

- [ ] **Step 3: Copy MusicFM source files into vendor**

```bash
cp src/third_party/musicfm/model/__init__.py songformer/vendor/musicfm/model/
cp src/third_party/musicfm/model/musicfm_25hz.py songformer/vendor/musicfm/model/
cp src/third_party/musicfm/modules/__init__.py songformer/vendor/musicfm/modules/
cp src/third_party/musicfm/modules/conv.py songformer/vendor/musicfm/modules/
cp src/third_party/musicfm/modules/features.py songformer/vendor/musicfm/modules/
cp src/third_party/musicfm/modules/random_quantizer.py songformer/vendor/musicfm/modules/
```

Note: We intentionally skip `modules/flash_conformer.py` — it's only used when `is_flash=True`, and SongFormer always passes `is_flash=False` to MusicFM.

- [ ] **Step 4: Create `__init__.py` files for the vendor packages**

Create `songformer/__init__.py` (empty placeholder for now, will be filled in Task 5):
```python
```

Create `songformer/vendor/__init__.py`:
```python
```

Create `songformer/vendor/musicfm/__init__.py`:
```python
```

- [ ] **Step 5: Fix MusicFM internal imports to use vendored paths**

The original MusicFM code uses `from musicfm.modules.X import Y` internally. These need to become `from songformer.vendor.musicfm.modules.X import Y`.

Edit `songformer/vendor/musicfm/model/musicfm_25hz.py` — change these three import lines:

```python
# OLD:
from musicfm.modules.random_quantizer import RandomProjectionQuantizer
from musicfm.modules.features import MelSTFT
from musicfm.modules.conv import Conv2dSubsampling

# NEW:
from songformer.vendor.musicfm.modules.random_quantizer import RandomProjectionQuantizer
from songformer.vendor.musicfm.modules.features import MelSTFT
from songformer.vendor.musicfm.modules.conv import Conv2dSubsampling
```

Also fix the conditional conformer import inside `__init__` of `MusicFM25Hz`. Find this block:

```python
if is_flash:
    from modules.flash_conformer import (
        Wav2Vec2ConformerEncoder,
        Wav2Vec2ConformerConfig,
    )
else:
    from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
        Wav2Vec2ConformerEncoder,
    )
    from transformers.models.wav2vec2_conformer.configuration_wav2vec2_conformer import (
        Wav2Vec2ConformerConfig,
    )
```

Change the `is_flash=True` branch to:

```python
if is_flash:
    from songformer.vendor.musicfm.modules.flash_conformer import (
        Wav2Vec2ConformerEncoder,
        Wav2Vec2ConformerConfig,
    )
```

(This branch won't be hit since we always use `is_flash=False`, but fixing it prevents confusion.)

- [ ] **Step 6: Verify the vendored module imports correctly**

Run: `python -c "from songformer.vendor.musicfm.model.musicfm_25hz import MusicFM25Hz; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add songformer/
git commit -m "feat: vendor MusicFM package into songformer/vendor/"
```

---

### Task 3: Make flash attention configurable in existing model code

**Files:**
- Modify: `src/SongFormer/models/SongFormer.py:40-68` (WrapedTransformerEncoder)
- Modify: `src/SongFormer/models/SongFormer.py:248-274` (Model.__init__)

- [ ] **Step 1: Add `attn_flash` parameter to `WrapedTransformerEncoder`**

In `src/SongFormer/models/SongFormer.py`, change the `WrapedTransformerEncoder.__init__` signature and the `Encoder` call:

```python
# OLD (line 41-68):
class WrapedTransformerEncoder(nn.Module):
    def __init__(
        self, input_dim, transformer_input_dim, num_layers=1, nhead=8, dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.transformer_input_dim = transformer_input_dim

        if input_dim != transformer_input_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, transformer_input_dim),
                nn.LayerNorm(transformer_input_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(transformer_input_dim, transformer_input_dim),
            )
        else:
            self.input_proj = nn.Identity()

        self.transformer = Encoder(
            dim=transformer_input_dim,
            depth=num_layers,
            heads=nhead,
            layer_dropout=dropout,
            attn_dropout=dropout,
            ff_dropout=dropout,
            attn_flash=True,
            rotary_pos_emb=True,
        )

# NEW:
class WrapedTransformerEncoder(nn.Module):
    def __init__(
        self, input_dim, transformer_input_dim, num_layers=1, nhead=8, dropout=0.1, attn_flash=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.transformer_input_dim = transformer_input_dim

        if input_dim != transformer_input_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, transformer_input_dim),
                nn.LayerNorm(transformer_input_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(transformer_input_dim, transformer_input_dim),
            )
        else:
            self.input_proj = nn.Identity()

        self.transformer = Encoder(
            dim=transformer_input_dim,
            depth=num_layers,
            heads=nhead,
            layer_dropout=dropout,
            attn_dropout=dropout,
            ff_dropout=dropout,
            attn_flash=attn_flash,
            rotary_pos_emb=True,
        )
```

- [ ] **Step 2: Pass `attn_flash` through `Model.__init__`**

In `src/SongFormer/models/SongFormer.py`, change the `Model.__init__` constructor:

```python
# OLD (line 248-274):
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ... (other init lines stay the same) ...
        self.transformer = WrapedTransformerEncoder(
            input_dim=config.transformer_encoder_input_dim,
            transformer_input_dim=config.transformer_input_dim,
            num_layers=config.num_transformer_layers,
            nhead=config.transformer_nhead,
            dropout=config.transformer_dropout,
        )

# NEW:
class Model(nn.Module):
    def __init__(self, config, attn_flash=True):
        super().__init__()
        self.config = config
        # ... (other init lines stay the same) ...
        self.transformer = WrapedTransformerEncoder(
            input_dim=config.transformer_encoder_input_dim,
            transformer_input_dim=config.transformer_input_dim,
            num_layers=config.num_transformer_layers,
            nhead=config.transformer_nhead,
            dropout=config.transformer_dropout,
            attn_flash=attn_flash,
        )
```

Only the constructor signature and the `WrapedTransformerEncoder(...)` call change. All other lines in `Model.__init__` remain identical.

- [ ] **Step 3: Commit**

```bash
git add src/SongFormer/models/SongFormer.py
git commit -m "feat: make attn_flash configurable in WrapedTransformerEncoder and Model"
```

---

### Task 4: Create Pydantic types

**Files:**
- Create: `songformer/types.py`

- [ ] **Step 1: Write the test**

Create `tests/test_types.py`:

```python
from songformer.types import Segment, AnalysisResult


def test_segment_creation():
    seg = Segment(start=0.0, end=15.3, label="intro")
    assert seg.start == 0.0
    assert seg.end == 15.3
    assert seg.label == "intro"


def test_analysis_result_creation():
    segments = [
        Segment(start=0.0, end=15.3, label="intro"),
        Segment(start=15.3, end=45.7, label="verse"),
    ]
    result = AnalysisResult(segments=segments, duration=90.0)
    assert len(result.segments) == 2
    assert result.duration == 90.0


def test_analysis_result_json_roundtrip():
    segments = [
        Segment(start=0.0, end=15.3, label="intro"),
        Segment(start=15.3, end=45.7, label="verse"),
    ]
    result = AnalysisResult(segments=segments, duration=90.0)
    json_str = result.model_dump_json()
    restored = AnalysisResult.model_validate_json(json_str)
    assert restored == result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'songformer.types'`

- [ ] **Step 3: Create `songformer/types.py`**

```python
from pydantic import BaseModel


class Segment(BaseModel):
    """A labeled section of a song."""
    start: float
    end: float
    label: str


class AnalysisResult(BaseModel):
    """Result of music structure analysis."""
    segments: list[Segment]
    duration: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add songformer/types.py tests/test_types.py
git commit -m "feat: add Segment and AnalysisResult Pydantic models"
```

---

### Task 5: Create the weight download module

**Files:**
- Create: `songformer/download.py`
- Create: `tests/test_download.py`

- [ ] **Step 1: Write the test**

Create `tests/test_download.py`:

```python
from unittest.mock import patch, call
from songformer.download import download_weights, WEIGHT_SPECS


def test_weight_specs_defined():
    """All three model weight sets are defined."""
    repos = [spec["repo_id"] for spec in WEIGHT_SPECS]
    assert "minzwon/MusicFM" in repos
    assert "ASLP-lab/SongFormer" in repos


def test_download_calls_hf_hub(tmp_path):
    """download_weights calls hf_hub_download for each weight file."""
    with patch("songformer.download.hf_hub_download") as mock_dl:
        mock_dl.return_value = str(tmp_path / "dummy.pt")
        download_weights()

    # MusicFM has 2 files, SongFormer has 1 = at least 3 calls
    assert mock_dl.call_count >= 3

    # Check that the repo_ids are correct
    called_repos = [c.kwargs["repo_id"] for c in mock_dl.call_args_list]
    assert "minzwon/MusicFM" in called_repos
    assert "ASLP-lab/SongFormer" in called_repos
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'songformer.download'`

- [ ] **Step 3: Create `songformer/download.py`**

```python
"""Download SongFormer model weights from HuggingFace Hub."""

from huggingface_hub import hf_hub_download


WEIGHT_SPECS = [
    {
        "repo_id": "minzwon/MusicFM",
        "filename": "msd_stats.json",
    },
    {
        "repo_id": "minzwon/MusicFM",
        "filename": "pretrained_msd.pt",
    },
    {
        "repo_id": "ASLP-lab/SongFormer",
        "filename": "SongFormer.safetensors",
    },
]


def download_weights() -> dict[str, str]:
    """Download all required model weights.

    Returns a dict mapping weight names to their cached file paths.
    MuQ weights are handled by the muq package's own from_pretrained().
    """
    paths: dict[str, str] = {}
    for spec in WEIGHT_SPECS:
        repo_id = spec["repo_id"]
        filename = spec["filename"]
        print(f"Downloading {repo_id}/{filename}...")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        paths[filename] = path
        print(f"  -> {path}")
    return paths


def get_weight_path(repo_id: str, filename: str) -> str:
    """Get the cached path for a weight file.

    Raises FileNotFoundError if weights haven't been downloaded.
    Run `songformer-download` first.
    """
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
    except Exception:
        raise FileNotFoundError(
            f"Weight file '{filename}' from '{repo_id}' not found in cache. "
            f"Run `songformer-download` to download all required weights."
        )


def main() -> None:
    """CLI entry point for songformer-download."""
    print("Downloading SongFormer model weights...")
    print("(MuQ weights will be downloaded on first pipeline load)\n")
    paths = download_weights()
    print(f"\nDone. {len(paths)} files downloaded/verified.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_download.py -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add songformer/download.py tests/test_download.py
git commit -m "feat: add weight download module with HuggingFace Hub integration"
```

---

### Task 6: Create the SongFormerPipeline class

**Files:**
- Create: `songformer/pipeline.py`
- Create: `tests/test_pipeline.py`

This is the core task — wrapping `app.py`'s `initialize_models()` and `process_audio()` into a class.

- [ ] **Step 1: Write the test**

Create `tests/test_pipeline.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

from songformer.pipeline import SongFormerPipeline, _detect_device


def test_detect_device_cpu():
    """Falls back to CPU when nothing else is available."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        assert _detect_device() == "cpu"


def test_detect_device_mps():
    """Prefers MPS on Apple Silicon."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True):
        assert _detect_device() == "mps"


def test_detect_device_cuda():
    """Prefers CUDA when available."""
    with patch("torch.cuda.is_available", return_value=True):
        assert _detect_device() == "cuda"


def test_flash_attention_defaults():
    """Flash attention auto-set based on device."""
    assert SongFormerPipeline._resolve_flash_attention(None, "cuda") is True
    assert SongFormerPipeline._resolve_flash_attention(None, "cpu") is False
    assert SongFormerPipeline._resolve_flash_attention(None, "mps") is False
    # Explicit override
    assert SongFormerPipeline._resolve_flash_attention(True, "cpu") is True
    assert SongFormerPipeline._resolve_flash_attention(False, "cuda") is False


def test_msa_info_to_segments():
    """Converts SongFormer's MsaInfo format to Segment list."""
    from songformer.types import Segment
    msa_info = [
        (0.0, "intro"),
        (15.3, "verse"),
        (45.7, "chorus"),
        (90.0, "end"),
    ]
    segments = SongFormerPipeline._msa_info_to_segments(msa_info)
    assert len(segments) == 3
    assert segments[0] == Segment(start=0.0, end=15.3, label="intro")
    assert segments[1] == Segment(start=15.3, end=45.7, label="verse")
    assert segments[2] == Segment(start=45.7, end=90.0, label="chorus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'songformer.pipeline'`

- [ ] **Step 3: Create `songformer/pipeline.py`**

```python
"""SongFormerPipeline — clean API for music structure analysis."""

from __future__ import annotations

import math
import sys
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf
from ema_pytorch import EMA

from songformer.types import Segment, AnalysisResult
from songformer.download import get_weight_path

# Constants from the original app.py
_INPUT_SR = 24000
_FRAME_RATE_AFTER_DS = 8.333
_TIME_DUR = 420
_DATASET_LABEL = "SongForm-HX-8Class"
_DATASET_ID = 5
_NUM_CLASSES = 128


def _detect_device() -> str:
    """Auto-detect best available device: mps → cuda → cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _setup_src_imports() -> None:
    """Add src/SongFormer/ to sys.path so internal imports work.

    Also applies the scipy.inf monkey-patch needed because
    src/SongFormer/models/SongFormer.py imports msaf at module level,
    and msaf uses the removed scipy.inf attribute.
    """
    repo_root = Path(__file__).resolve().parent.parent
    songformer_src = str(repo_root / "src" / "SongFormer")
    if songformer_src not in sys.path:
        sys.path.insert(0, songformer_src)
        # Monkey-patch scipy.inf before any import that touches msaf
        import scipy
        scipy.inf = np.inf  # type: ignore[attr-defined]


def _load_config() -> object:
    """Load the SongFormer YAML config."""
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "src" / "SongFormer" / "configs" / "SongFormer.yaml"
    return OmegaConf.load(str(config_path))


class SongFormerPipeline:
    """Music structure analysis pipeline using SongFormer."""

    def __init__(
        self,
        muq_model: torch.nn.Module,
        musicfm_model: torch.nn.Module,
        msa_model: torch.nn.Module,
        config: object,
        device: torch.device,
    ):
        self.muq = muq_model
        self.musicfm = musicfm_model
        self.msa = msa_model
        self.config = config
        self.device = device

    @staticmethod
    def _resolve_flash_attention(flash_attention: bool | None, device: str) -> bool:
        if flash_attention is not None:
            return flash_attention
        return device == "cuda"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        flash_attention: bool | None = None,
    ) -> SongFormerPipeline:
        """Load all models from pre-downloaded weights.

        Args:
            device: Target device ("cpu", "cuda", "mps"). Auto-detects if None.
            flash_attention: Use flash attention in transformer. Auto-set based
                on device if None (True for CUDA, False otherwise).

        Raises:
            FileNotFoundError: If weights haven't been downloaded yet.
                Run `songformer-download` first.
        """
        if device is None:
            device = _detect_device()

        attn_flash = cls._resolve_flash_attention(flash_attention, device)
        torch_device = torch.device(device)

        _setup_src_imports()

        # --- Load MuQ ---
        from muq import MuQ

        muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        muq_model = muq_model.to(torch_device).eval()

        # --- Load MusicFM ---
        from songformer.vendor.musicfm.model.musicfm_25hz import MusicFM25Hz

        stat_path = get_weight_path("minzwon/MusicFM", "msd_stats.json")
        model_path = get_weight_path("minzwon/MusicFM", "pretrained_msd.pt")
        musicfm_model = MusicFM25Hz(
            is_flash=False,
            stat_path=stat_path,
            model_path=model_path,
        )
        musicfm_model = musicfm_model.to(torch_device).eval()

        # --- Load SongFormer MSA model ---
        import importlib

        module = importlib.import_module("models.SongFormer")
        Model = getattr(module, "Model")
        config = _load_config()
        msa_model = Model(config, attn_flash=attn_flash)

        # Load checkpoint with EMA unwrapping
        ckpt_path = get_weight_path("ASLP-lab/SongFormer", "SongFormer.safetensors")
        from safetensors.torch import load_file

        checkpoint = {"model_ema": load_file(ckpt_path, device=device)}

        model_ema = EMA(msa_model, include_online_model=False)
        model_ema.load_state_dict(checkpoint["model_ema"])
        msa_model.load_state_dict(model_ema.ema_model.state_dict())
        msa_model.to(torch_device).eval()

        return cls(
            muq_model=muq_model,
            musicfm_model=musicfm_model,
            msa_model=msa_model,
            config=config,
            device=torch_device,
        )

    def analyze(self, audio_path: str) -> AnalysisResult:
        """Analyze an audio file and return labeled song sections.

        Args:
            audio_path: Path to an audio file (any format librosa supports).

        Returns:
            AnalysisResult with segments and duration.
        """
        _setup_src_imports()
        from postprocessing.functional import postprocess_functional_structure
        from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID

        # Load audio
        wav, _ = librosa.load(audio_path, sr=_INPUT_SR)
        audio = torch.tensor(wav).to(self.device)
        duration = len(wav) / _INPUT_SR

        # Prepare logit accumulators
        total_len = (int(audio.shape[0] // _INPUT_SR) // _TIME_DUR * _TIME_DUR) + _TIME_DUR
        total_frames = math.ceil(total_len * _FRAME_RATE_AFTER_DS)

        logits = {
            "function_logits": np.zeros([total_frames, _NUM_CLASSES]),
            "boundary_logits": np.zeros([total_frames]),
        }
        logits_num = {
            "function_logits": np.zeros([total_frames, _NUM_CLASSES]),
            "boundary_logits": np.zeros([total_frames]),
        }

        # Prepare label masks
        dataset_id2label_mask: dict[int, np.ndarray] = {}
        for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
            dataset_id2label_mask[key] = np.ones(_NUM_CLASSES, dtype=bool)
            dataset_id2label_mask[key][allowed_ids] = False

        lens = 0
        i = 0

        with torch.no_grad():
            while True:
                start_idx = i * _INPUT_SR
                end_idx = min((i + _TIME_DUR) * _INPUT_SR, audio.shape[-1])
                if start_idx >= audio.shape[-1]:
                    break
                if end_idx - start_idx <= 1024:
                    break

                audio_seg = audio[start_idx:end_idx]

                # 420s embeddings
                muq_output = self.muq(audio_seg.unsqueeze(0), output_hidden_states=True)
                muq_embd_420s = muq_output["hidden_states"][10]
                del muq_output
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                _, musicfm_hidden_states = self.musicfm.get_predictions(
                    audio_seg.unsqueeze(0)
                )
                musicfm_embd_420s = musicfm_hidden_states[10]
                del musicfm_hidden_states
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                # 30s sub-window embeddings
                wrapped_muq_30s = []
                wrapped_musicfm_30s = []

                for idx_30s in range(i, i + _TIME_DUR, 30):
                    s = idx_30s * _INPUT_SR
                    e = min((idx_30s + 30) * _INPUT_SR, audio.shape[-1], (i + _TIME_DUR) * _INPUT_SR)
                    if s >= audio.shape[-1]:
                        break
                    if e - s <= 1024:
                        continue

                    wrapped_muq_30s.append(
                        self.muq(audio[s:e].unsqueeze(0), output_hidden_states=True)["hidden_states"][10]
                    )
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                    wrapped_musicfm_30s.append(
                        self.musicfm.get_predictions(audio[s:e].unsqueeze(0))[1][10]
                    )
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                if wrapped_muq_30s:
                    wrapped_muq_30s_cat = torch.concatenate(wrapped_muq_30s, dim=1)
                    wrapped_musicfm_30s_cat = torch.concatenate(wrapped_musicfm_30s, dim=1)

                    all_embds = [
                        wrapped_musicfm_30s_cat,
                        wrapped_muq_30s_cat,
                        musicfm_embd_420s,
                        muq_embd_420s,
                    ]

                    # Align lengths
                    min_len = min(x.shape[1] for x in all_embds)
                    all_embds = [x[:, :min_len, :] for x in all_embds]
                    embd = torch.concatenate(all_embds, axis=-1)

                    # Run model inference
                    dataset_ids = torch.tensor([_DATASET_ID], device=self.device, dtype=torch.long)
                    label_mask = torch.tensor(
                        dataset_id2label_mask[DATASET_LABEL_TO_DATASET_ID[_DATASET_LABEL]],
                        device=self.device,
                        dtype=torch.bool,
                    ).unsqueeze(0).unsqueeze(0)

                    _, chunk_logits = self.msa.infer(
                        input_embeddings=embd,
                        dataset_ids=dataset_ids,
                        label_id_masks=label_mask,
                        with_logits=True,
                    )

                    # Accumulate logits
                    start_frame = int(i * _FRAME_RATE_AFTER_DS)
                    end_frame = start_frame + min(
                        math.ceil(_TIME_DUR * _FRAME_RATE_AFTER_DS),
                        chunk_logits["boundary_logits"][0].shape[0],
                    )

                    logits["function_logits"][start_frame:end_frame, :] += (
                        chunk_logits["function_logits"][0].detach().cpu().numpy()
                    )
                    logits["boundary_logits"][start_frame:end_frame] = (
                        chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                    )
                    logits_num["function_logits"][start_frame:end_frame, :] += 1
                    logits_num["boundary_logits"][start_frame:end_frame] += 1
                    lens += end_frame - start_frame

                i += _TIME_DUR

        # Average and convert to tensors
        logits["function_logits"] /= np.maximum(logits_num["function_logits"], 1)
        logits["boundary_logits"] /= np.maximum(logits_num["boundary_logits"], 1)
        logits["function_logits"] = torch.from_numpy(logits["function_logits"][:lens]).unsqueeze(0)
        logits["boundary_logits"] = torch.from_numpy(logits["boundary_logits"][:lens]).unsqueeze(0)

        # Post-process
        msa_output = postprocess_functional_structure(logits, self.config)
        msa_output = self._rule_post_processing(msa_output)

        segments = self._msa_info_to_segments(msa_output)
        return AnalysisResult(segments=segments, duration=duration)

    @staticmethod
    def _msa_info_to_segments(msa_info: list[tuple[float, str]]) -> list[Segment]:
        """Convert SongFormer's MsaInfo to a list of Segment."""
        segments = []
        for i in range(len(msa_info) - 1):
            start_time, label = msa_info[i]
            end_time = msa_info[i + 1][0]
            if label != "end":
                segments.append(Segment(
                    start=round(start_time, 2),
                    end=round(end_time, 2),
                    label=label,
                ))
        return segments

    @staticmethod
    def _rule_post_processing(msa_list: list[tuple[float, str]]) -> list[tuple[float, str]]:
        """Rule-based cleanup: remove very short segments, merge adjacent same-label segments."""
        if len(msa_list) <= 2:
            return msa_list

        result = msa_list.copy()

        # Remove short first segment
        while len(result) > 2:
            first_duration = result[1][0] - result[0][0]
            if first_duration < 1.0 and len(result) > 2:
                result[0] = (result[0][0], result[1][1])
                result = [result[0]] + result[2:]
            else:
                break

        # Remove short last segment
        while len(result) > 2:
            last_label_duration = result[-1][0] - result[-2][0]
            if last_label_duration < 1.0:
                result = result[:-2] + [result[-1]]
            else:
                break

        # Merge same labels at start
        while len(result) > 2:
            if result[0][1] == result[1][1] and result[1][0] <= 10.0:
                result = [(result[0][0], result[0][1])] + result[2:]
            else:
                break

        # Merge same labels at end
        while len(result) > 2:
            last_duration = result[-1][0] - result[-2][0]
            if result[-2][1] == result[-3][1] and last_duration <= 10.0:
                result = result[:-2] + [result[-1]]
            else:
                break

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add songformer/pipeline.py tests/test_pipeline.py
git commit -m "feat: add SongFormerPipeline with analyze() method"
```

---

### Task 7: Wire up `__init__.py` and verify public API

**Files:**
- Modify: `songformer/__init__.py`

- [ ] **Step 1: Write the test**

Create `tests/test_public_api.py`:

```python
def test_public_imports():
    """All public symbols are importable from the top-level package."""
    from songformer import SongFormerPipeline, Segment, AnalysisResult
    assert SongFormerPipeline is not None
    assert Segment is not None
    assert AnalysisResult is not None


def test_no_side_effects_on_import():
    """Importing songformer does not trigger model loading or sys.path changes."""
    import sys
    path_before = sys.path.copy()
    import songformer  # noqa: F401
    # sys.path should not have been modified by the import
    assert sys.path == path_before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_public_api.py -v`
Expected: FAIL — `ImportError: cannot import name 'SongFormerPipeline' from 'songformer'`

- [ ] **Step 3: Update `songformer/__init__.py`**

```python
"""SongFormer — music structure analysis library."""

from songformer.types import Segment, AnalysisResult
from songformer.pipeline import SongFormerPipeline

__all__ = ["SongFormerPipeline", "Segment", "AnalysisResult"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_public_api.py -v`
Expected: Both tests PASS

- [ ] **Step 5: Run all tests together**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add songformer/__init__.py tests/test_public_api.py
git commit -m "feat: wire up public API exports in __init__.py"
```

---

### Task 8: End-to-end smoke test with real weights

This task verifies the full pipeline works on actual hardware. It requires weights to be downloaded first.

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Download weights**

Run: `uv run songformer-download`
Expected: Three files downloaded (or confirmed cached):
- `msd_stats.json`
- `pretrained_msd.pt`
- `SongFormer.safetensors`

- [ ] **Step 2: Write the smoke test**

Create `tests/test_e2e.py`:

```python
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from songformer import SongFormerPipeline, AnalysisResult


@pytest.mark.slow
def test_analyze_synthetic_audio(tmp_path):
    """Full pipeline smoke test with a synthetic audio file."""
    # Generate 60 seconds of synthetic audio at 24kHz
    sr = 24000
    duration = 60
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)

    # First 30s: simple sine wave (simulates a "section")
    # Last 30s: different frequency (simulates section change)
    audio = np.concatenate([
        0.5 * np.sin(2 * np.pi * 440 * t[:sr * 30]),   # A4
        0.5 * np.sin(2 * np.pi * 880 * t[:sr * 30]),   # A5
    ])

    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), audio, sr)

    # Run pipeline
    pipeline = SongFormerPipeline.from_pretrained()
    result = pipeline.analyze(str(audio_path))

    # Verify output structure
    assert isinstance(result, AnalysisResult)
    assert result.duration == pytest.approx(60.0, abs=1.0)
    assert len(result.segments) >= 1

    # Each segment has valid fields
    for seg in result.segments:
        assert seg.start >= 0.0
        assert seg.end > seg.start
        assert isinstance(seg.label, str)
        assert len(seg.label) > 0

    # Segments are contiguous
    for i in range(len(result.segments) - 1):
        assert result.segments[i].end == pytest.approx(result.segments[i + 1].start, abs=0.01)
```

- [ ] **Step 3: Run the slow test**

Run: `uv run pytest tests/test_e2e.py -v -m slow`
Expected: PASS — the pipeline loads models, processes synthetic audio, and returns valid segments.

If it fails, debug based on the error:
- `FileNotFoundError` → weights not downloaded, re-run `songformer-download`
- `RuntimeError` with flash attention → the `attn_flash` parameter isn't being passed through correctly (check Task 3)
- Import errors → `sys.path` setup in `_setup_src_imports()` isn't finding `src/SongFormer/`

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: add end-to-end smoke test with synthetic audio"
```