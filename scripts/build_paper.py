#!/usr/bin/env python
"""
Build the manuscript PDF from tmlr_draft.md (Markdown source of truth).

pandoc drops raw HTML when targeting LaTeX/PDF, so the ``<p align=center><img
src=../figures/..></p>`` blocks in the draft are first rewritten to pandoc image
syntax; math is passed through with the tex_math_single_backslash extension so
``\\(...\\)`` / ``\\[...\\]`` render. PDF is produced by tectonic (self-contained
LaTeX). Invoked by ``make paper``.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tmlr_draft.md"
BUILD = ROOT / ".build"
PRE = BUILD / "tmlr_draft.pandoc.md"
OUT = ROOT / "tmlr_draft.pdf"
TECTONIC = BUILD / "bin" / "tectonic"


def ensure_pandoc():
    """Ensure a pandoc binary is available (bundled via pypandoc-binary)."""
    try:
        import pypandoc
        pypandoc.get_pandoc_path()
        return
    except Exception:
        print("[build] installing pandoc (pypandoc-binary) ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypandoc-binary"],
                       check=True)


def ensure_tectonic():
    """Download the self-contained tectonic LaTeX engine for this arch if missing."""
    if TECTONIC.exists():
        return
    arch = {"aarch64": "aarch64", "arm64": "aarch64", "x86_64": "x86_64"}.get(platform.machine())
    if arch is None:
        sys.exit(f"unsupported arch {platform.machine()!r}; install tectonic manually into {TECTONIC}")
    print(f"[build] downloading tectonic ({arch}) ...")
    api = "https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest"
    rel = json.loads(urllib.request.urlopen(api, timeout=30).read())
    asset = next(a["browser_download_url"] for a in rel["assets"]
                 if f"{arch}-unknown-linux-musl.tar.gz" in a["name"])
    (BUILD / "bin").mkdir(parents=True, exist_ok=True)
    tgz = BUILD / "tectonic.tar.gz"
    urllib.request.urlretrieve(asset, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(BUILD / "bin")
    TECTONIC.chmod(0o755)

# <p align="center"> ... <img src="../figures/X.png" ...> ... </p>  ->  ![](figures/X.png){width=72%}
_FIG = re.compile(
    r'<p align="center">\s*<img\s+src="\.\./(figures/[^"]+)"[^>]*?/?>\s*</p>',
    re.DOTALL,
)


def preprocess(text: str) -> str:
    text = _FIG.sub(lambda m: f'\n![]({m.group(1)}){{width=72%}}\n', text)
    # drop any stray <sub>/<br> raw-HTML that pandoc would warn on in captions
    text = re.sub(r'</?sub>', '', text)
    text = text.replace('<br/>', '  ')
    # angle brackets are absent from most text fonts; map to math so they always render
    text = text.replace('⟨', r'\(\langle\)').replace('⟩', r'\(\rangle\)')
    return text


def main() -> int:
    if not SRC.exists():
        sys.exit(f"missing manuscript source: {SRC}")
    BUILD.mkdir(exist_ok=True)
    ensure_pandoc()
    ensure_tectonic()
    PRE.write_text(preprocess(SRC.read_text()))

    import pypandoc
    pandoc = pypandoc.get_pandoc_path()
    cmd = [
        pandoc, str(PRE), "-o", str(OUT),
        "--pdf-engine", str(TECTONIC),
        "-f", "markdown+tex_math_single_backslash",
        "--toc", "--resource-path", str(ROOT),
        "-V", "geometry:margin=1in", "-V", "fontsize=11pt",
        "-V", "mainfont=FreeSerif",   # Unicode coverage (kappa/sigma/subscripts/~) via XeTeX
        "-V", "linkcolor:blue", "-V", "urlcolor:blue",
        "--metadata", "title=Learned Representations of Non-Local Quantum Order in Energy-Predicting Transformers",
    ]
    print("building:", " ".join(cmd[:1]), "... -> ", OUT.name)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        sys.exit(f"pandoc/tectonic failed (exit {r.returncode})")
    kb = OUT.stat().st_size / 1024
    print(f"OK -> {OUT}  ({kb:.0f} KB)")
    missing = [ln for ln in r.stderr.splitlines() if "Missing character" in ln or "could not represent" in ln]
    if missing:
        print(f"--- WARNING: {len(missing)} unrenderable characters (glyph gaps) ---")
        print("\n".join(missing[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
