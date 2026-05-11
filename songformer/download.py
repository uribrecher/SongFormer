"""Pre-fetch the model weights SongFormerPipeline needs into the HuggingFace cache.

Usage:
    songformer-download
    python -m songformer.download
"""

from __future__ import annotations

import sys

from huggingface_hub import hf_hub_download

_DOWNLOADS = [
    ("minzwon/MusicFM", "msd_stats.json"),
    ("minzwon/MusicFM", "pretrained_msd.pt"),
    ("ASLP-lab/SongFormer", "SongFormer.safetensors"),
]


def main(argv: list[str] | None = None) -> int:
    del argv
    print("Downloading SongFormer weights into the HuggingFace cache...")
    for repo_id, filename in _DOWNLOADS:
        print(f"  {repo_id}/{filename}")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"    → {path}")

    # MuQ pulls its own weights on first `MuQ.from_pretrained(...)`.
    print(
        "MuQ weights will be fetched on first SongFormerPipeline.from_pretrained() "
        "call — the `muq` package handles that itself."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
