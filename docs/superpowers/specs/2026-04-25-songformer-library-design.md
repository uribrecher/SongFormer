# SongFormer Library Design

Turn the SongFormer fork (https://github.com/uribrecher/SongFormer) into an installable Python library with a clean API for music structure analysis.

## Context

SongFormer (ASLP-lab, Oct 2025) is the current SOTA for music structure analysis. It works directly on mixed audio (no stem separation required), outputs labeled sections with timestamps, and uses self-supervised embeddings (MuQ + MusicFM) as input features.

The upstream repo is research code — global mutable state, `os.chdir()` / `sys.path` hacks, hardcoded CUDA assumptions, no `pyproject.toml`. The goal is to wrap it in a proper Python package without restructuring the internals.

## Repository structure

```
SongFormer/                        ← forked repo root
├── pyproject.toml                 ← NEW: package metadata + deps
├── songformer/                    ← NEW: public API package
│   ├── __init__.py                ← exports SongFormerPipeline, Segment, AnalysisResult
│   ├── pipeline.py                ← SongFormerPipeline class
│   ├── download.py                ← weight download CLI
│   ├── types.py                   ← Segment, AnalysisResult (Pydantic models)
│   └── vendor/
│       └── musicfm/               ← MusicFM vendored (was git submodule)
│           ├── model/
│           └── modules/
├── src/
│   ├── SongFormer/                ← existing code, mostly untouched
│   │   ├── app.py
│   │   ├── models/
│   │   ├── postprocessing/
│   │   ├── dataset/
│   │   ├── configs/
│   │   ├── ckpts/                 ← gitignored, weights go to HuggingFace cache instead
│   │   └── infer/
│   └── third_party/musicfm/      ← original submodule, now unused
└── ...
```

## Public API

```python
from songformer import SongFormerPipeline, Segment, AnalysisResult

pipeline = SongFormerPipeline.from_pretrained(device="mps")
result = pipeline.analyze("song.mp3")

for seg in result.segments:
    print(f"{seg.start:.1f}–{seg.end:.1f}  {seg.label}")
```

## SongFormerPipeline class

```python
class SongFormerPipeline:
    def __init__(self, muq_model, musicfm_model, msa_model, config, device):
        self.muq = muq_model
        self.musicfm = musicfm_model
        self.msa = msa_model
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,            # None = auto: mps → cuda → cpu
        flash_attention: bool | None = None,   # None = auto: True if CUDA
    ) -> "SongFormerPipeline":
        ...

    def analyze(self, audio_path: str) -> AnalysisResult:
        ...
```

**Device auto-detection:** `mps` → `cuda` → `cpu`.

**Flash attention:** defaults to `True` on CUDA, `False` on CPU/MPS. The `WrapedTransformerEncoder` in `src/SongFormer/models/SongFormer.py` is modified to accept `attn_flash` as a constructor parameter instead of hardcoding `True`.

**Weight loading:** `from_pretrained()` loads weights from the HuggingFace cache (`~/.cache/huggingface/`). If weights are not found, it raises a clear error directing the user to run the download command. No implicit network calls.

## Types (Pydantic models)

```python
from pydantic import BaseModel

class Segment(BaseModel):
    start: float      # seconds
    end: float        # seconds
    label: str        # "intro", "verse", "pre-chorus", "chorus", "bridge", "inst", "outro", "silence"

class AnalysisResult(BaseModel):
    segments: list[Segment]
    duration: float   # total audio duration in seconds
```

## Weight download

Two interfaces for downloading model weights:

1. **CLI entry point:** `songformer-download` (defined in `pyproject.toml` `[project.scripts]`)
2. **Module:** `python -m songformer.download`

Both download all three sets of weights via `huggingface_hub.hf_hub_download()`:

| Weight | HuggingFace source |
|--------|--------------------|
| MuQ | `OpenMuQ/MuQ-large-msd-iter` (auto-handled by `muq` package) |
| MusicFM | `minzwon/MusicFM` → `pretrained_msd.pt`, `msd_stats.json` |
| SongFormer | `ASLP-lab/SongFormer` → `SongFormer.safetensors` |

Weights are cached in `~/.cache/huggingface/` (the standard HuggingFace cache). The download command is idempotent — skips already-cached files.

## Vendoring MusicFM

MusicFM is not on PyPI and has no `pyproject.toml`. The `musicfm/` Python package (from inside the git submodule at `src/third_party/musicfm/`) is copied into `songformer/vendor/musicfm/`.

The pipeline adds `songformer/vendor/` to `sys.path` before importing MusicFM. This is the same hack the original code uses, contained within the `songformer` package.

The hardcoded `ckpts/MusicFM/` paths in the original code are replaced — the pipeline resolves weight paths via `huggingface_hub.hf_hub_download()` and passes them to the `MusicFM25Hz` constructor.

The original git submodule at `src/third_party/musicfm/` stays in the repo (not removed).

## Modifications to existing code

Minimal changes to `src/SongFormer/`:

1. **`models/SongFormer.py`** — `WrapedTransformerEncoder.__init__()` takes `attn_flash: bool` parameter instead of hardcoding `True`.
2. **`models/SongFormer.py`** — `Model.__init__()` passes `attn_flash` through to `WrapedTransformerEncoder`.

The `scipy.inf = np.inf` monkey-patch (needed for transitive msaf imports) is handled in `songformer/pipeline.py` if needed, not in the original code.

## pyproject.toml

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
]

[project.scripts]
songformer-download = "songformer.download:main"
```

Note: `triton` is excluded — it's CUDA-only and only needed for flash attention training, not inference.