"""Fonts and KaTeX for the page renderer: finding, fetching and @font-face CSS.

Fonts come from the google/fonts GitHub repo (OFL/Apache/UFL), the same
source ResumeOCR-Lite's fetch_fonts.py uses, so an existing ``fonts/`` folder
from that project works as-is. KaTeX (MIT) comes from the npm registry
tarball; its ``fonts/`` folder must sit next to ``katex.min.css``.
"""
from __future__ import annotations

import io
import os
import re
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

GF_BASE = "https://raw.githubusercontent.com/google/fonts/main"
LICENSE_DIRS = ("ofl", "apache", "ufl")
KATEX_VERSION = "0.16.22"

RESUME_FAMILIES = [
    "carlito", "caladea", "arimo", "tinos", "cousine", "gelasio", "ebgaramond",
    "lato", "roboto", "opensans", "sourcesans3", "sourceserif4", "montserrat", "raleway",
    "poppins", "merriweather", "lora", "inter", "notosans", "notoserif", "ibmplexsans",
    "ibmplexserif", "ptsans", "ptserif", "firasans", "nunito", "worksans", "ubuntu",
    "librebaskerville", "crimsontext", "robotoslab", "karla", "rubik", "mulish", "dmsans",
    "manrope", "barlow", "heebo", "hind", "titilliumweb", "cabin", "librefranklin",
    "publicsans", "spectral", "bitter", "domine", "oswald", "playfairdisplay",
    "robotomono", "sourcecodepro", "ibmplexmono",
]
EXTRA_FAMILIES = [
    "notosanssymbols2",           # arrows, check marks, geometric bullets
    "specialelite", "courierprime",  # typewriter faces for old-scan pages
    "notosansdevanagari", "notoserifdevanagari", "notosanstamil", "notosansbengali",
    "notosanstelugu", "notosanskannada", "notosansgujarati", "notosansgurmukhi", "notosansmalayalam",
]

SERIF = {"caladea", "tinos", "gelasio", "ebgaramond", "sourceserif4", "merriweather", "lora", "notoserif",
         "ibmplexserif", "ptserif", "librebaskerville", "crimsontext", "robotoslab", "spectral", "bitter",
         "domine", "playfairdisplay", "notoserifdevanagari"}
MONO = {"cousine", "robotomono", "sourcecodepro", "ibmplexmono", "courierprime", "specialelite"}
DISPLAY = {"oswald", "playfairdisplay"}
TYPEWRITER = {"courierprime", "specialelite", "cousine"}
SYMBOL = {"notosanssymbols2"}
INDIC = {"notosansdevanagari": "deva", "notoserifdevanagari": "deva", "notosanstamil": "taml",
         "notosansbengali": "beng", "notosanstelugu": "telu", "notosanskannada": "knda",
         "notosansgujarati": "gujr", "notosansgurmukhi": "guru", "notosansmalayalam": "mlym"}
_SKIP_WEIGHT = re.compile(r"Thin|Hairline|ExtraLight|UltraLight|Black|Heavy", re.I)


# ------------------------------------------------------------------ fetching
def _fetch(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Folio-OCR-asset-fetcher"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_font_family(family: str, dest: str) -> int:
    """Download one Google Fonts family into dest/<family>/. Returns files written."""
    for lic in LICENSE_DIRS:
        try:
            meta = _fetch(f"{GF_BASE}/{lic}/{family}/METADATA.pb").decode("utf-8", "replace")
        except urllib.error.HTTPError:
            continue
        names = sorted(set(re.findall(r'filename:\s*"([^"]+)"', meta)))
        names = [n for n in names if not _SKIP_WEIGHT.search(n)] or names
        out = Path(dest) / family
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in names:
            p = out / name
            if p.exists() and p.stat().st_size > 0:
                continue
            p.write_bytes(_fetch(f"{GF_BASE}/{lic}/{family}/{urllib.request.quote(name)}"))
            n += 1
        for lic_file in ("OFL.txt", "LICENSE.txt", "UFL.txt"):
            try:
                (out / lic_file).write_bytes(_fetch(f"{GF_BASE}/{lic}/{family}/{lic_file}"))
                break
            except urllib.error.HTTPError:
                continue
        return n
    raise FileNotFoundError(f"font family {family!r} not found in google/fonts")


def fetch_katex(dest: str, version: str = KATEX_VERSION) -> Path:
    """npm tarball -> dest/katex/{katex.min.js, katex.min.css, fonts/}."""
    out = Path(dest) / "katex"
    if (out / "katex.min.js").exists() and (out / "fonts").is_dir():
        return out
    data = _fetch(f"https://registry.npmjs.org/katex/-/katex-{version}.tgz")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile() or not m.name.startswith("package/dist/"):
                continue
            rel = m.name[len("package/dist/"):]
            if not (rel in ("katex.min.js", "katex.min.css") or rel.startswith("fonts/")):
                continue
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(m).read())
        try:
            (out / "LICENSE").write_bytes(tar.extractfile("package/LICENSE").read())
        except KeyError:
            pass
    return out


# ----------------------------------------------------------------- locating
def find_assets_dir(explicit: Optional[str] = None) -> Path:
    """Folder holding ``fonts/`` and ``katex/`` (FOLIO_ASSETS or ./assets)."""
    for cand in (explicit, os.environ.get("FOLIO_ASSETS"), "assets",
                 str(Path(__file__).resolve().parents[2] / "assets")):
        if cand and Path(cand).is_dir():
            return Path(cand).resolve()
    raise FileNotFoundError("assets folder not found: run `python fetch_assets.py` first "
                            "(or set FOLIO_ASSETS=/path/to/assets)")


# ------------------------------------------------------------------- fonts
@dataclass
class Face:
    path: str
    weight: str          # "400" or "100 900" (variable)
    italic: bool


@dataclass
class Family:
    name: str
    kind: str            # serif | sans | mono | display | symbol | indic
    faces: List[Face] = field(default_factory=list)
    script: str = "latn"

    @property
    def css_name(self) -> str:
        return f"F_{self.name}"

    def has_bold(self) -> bool:
        return any(f.weight == "700" or " " in f.weight for f in self.faces)

    def has_italic(self) -> bool:
        return any(f.italic for f in self.faces)


_STATIC_WEIGHTS = {"Regular": "400", "Italic": "400", "Bold": "700", "BoldItalic": "700",
                   "Medium": "500", "MediumItalic": "500", "SemiBold": "600", "SemiBoldItalic": "600"}


def _faces_for(dirpath: Path) -> List[Face]:
    faces = []
    for p in sorted(dirpath.glob("*.[ot]tf")):
        stem = p.stem
        if "[" in stem:                                     # variable font
            faces.append(Face(str(p), "100 900", "Italic" in stem.split("[")[0]))
            continue
        style = stem.split("-")[-1] if "-" in stem else "Regular"
        if style in _STATIC_WEIGHTS:
            faces.append(Face(str(p), _STATIC_WEIGHTS[style], "Italic" in style))
    # prefer a variable face over statics if both exist
    if any(" " in f.weight for f in faces):
        faces = [f for f in faces if " " in f.weight]
    return faces


@lru_cache(maxsize=4)
def load_families(fonts_dir: str) -> Tuple[Family, ...]:
    out = []
    for d in sorted(Path(fonts_dir).iterdir()):
        if not d.is_dir():
            continue
        faces = _faces_for(d)
        if not faces:
            continue
        name = d.name.lower()
        if name in SYMBOL:
            kind = "symbol"
        elif name in INDIC:
            kind = "indic"
        elif name in MONO:
            kind = "mono"
        elif name in DISPLAY:
            kind = "display"
        elif name in SERIF:
            kind = "serif"
        else:
            kind = "sans"
        out.append(Family(name, kind, faces, INDIC.get(name, "latn")))
    if not out:
        raise FileNotFoundError(f"no fonts found under {fonts_dir}")
    return tuple(out)


def font_face_css(fam: Family) -> str:
    rules = []
    for f in fam.faces:
        url = Path(f.path).resolve().as_uri()
        rules.append(f"@font-face{{font-family:'{fam.css_name}';src:url('{url}');"
                     f"font-weight:{f.weight};font-style:{'italic' if f.italic else 'normal'};font-display:block;}}")
    return "\n".join(rules)


@lru_cache(maxsize=256)
def font_coverage(path: str) -> frozenset:
    from fontTools.ttLib import TTFont
    try:
        f = TTFont(path, lazy=True, fontNumber=0)
        cmap = f.getBestCmap() or {}
        f.close()
        return frozenset(cmap.keys())
    except Exception:
        return frozenset()


def family_coverage(fam: Family) -> frozenset:
    cov = frozenset()
    for f in fam.faces:
        if not f.italic:
            cov = cov | font_coverage(f.path)
    return cov


def uncovered(text: str, fams: Iterable[Family]) -> Set[str]:
    cov = frozenset()
    for fam in fams:
        cov = cov | family_coverage(fam)
    return {c for c in set(text) if not c.isspace() and ord(c) not in cov}
