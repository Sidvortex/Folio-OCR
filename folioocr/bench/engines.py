"""OCR engines behind one interface: page images in, Markdown out.

    folio     our model (or plain Infinity-Parser2-Flash / any Qwen3.5-family
              checkpoint), run in-process with transformers
    chandra   Datalab's ``chandra`` CLI (pip install chandra-ocr[hf])
    surya     Datalab's ``surya_ocr`` CLI (pip install surya-ocr; spawns its own
              vLLM / llama.cpp server)
    command   any tool: a shell template with {input} and {output} folders that
              leaves <stem>.md files in {output}
    mddir     precomputed Markdown files (<key>.md), e.g. from a paid API

Every engine gets the same images when the harness renders pages itself
("common" setting); ``native_pdf`` engines can instead take the PDF and render
it their own way ("native" setting).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from ..convert import html_to_markdown, surya_page_to_markdown


@dataclass
class PageItem:
    key: str                         # unique, filesystem-safe
    image: Optional[str] = None      # rendered page image (PNG)
    pdf: Optional[str] = None        # source PDF (for native rendering)
    page: int = 1


def safe_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s)


class Engine:
    name = "engine"
    native_pdf = False               # can take PDFs directly

    def run(self, items: List[PageItem]) -> Dict[str, str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FolioEngine(Engine):
    name = "folio"

    def __init__(self, model: str, adapter: Optional[str] = None, task: str = "doc2json", max_pixels: int = 2_000_000,
                 dtype: str = "auto", load_4bit: bool = False, batch_size: int = 1, max_new_tokens: int = 8192,
                 second_look: bool = False, keep_headers: bool = False):
        from ..model.generate import PageParser
        from ..model.load import ModelConfig, load_model
        self.model, self.processor = load_model(ModelConfig(model=model, adapter=adapter, dtype=dtype, load_4bit=load_4bit))
        self.parser = PageParser(self.model, self.processor, task=task, max_pixels=max_pixels, batch_size=batch_size,
                                 max_new_tokens=max_new_tokens, keep_non_body=keep_headers)
        self.second_look = second_look
        self.stats: List[dict] = []

    def run(self, items: List[PageItem]) -> Dict[str, str]:
        out = {}
        bs = self.parser.batch_size
        for s in range(0, len(items), bs):
            chunk = items[s:s + bs]
            imgs = [Image.open(it.image).convert("RGB") for it in chunk]
            results = self.parser.parse(imgs)
            if self.second_look:
                from ..infer import second_look
                for img, r in zip(imgs, results):
                    second_look(self.parser, img, r)
            for it, r in zip(chunk, results):
                out[it.key] = r.markdown
                self.stats.append({"key": it.key, "tokens": r.tokens, "seconds": r.seconds, "looped": r.looped,
                                   "retried": r.retried, "parse_ok": r.parse_ok})
        return out


class _FolderCLI(Engine):
    """Base for CLIs that take an input folder and write an output folder."""
    native_pdf = True

    def _stage(self, items: List[PageItem], use_pdf: bool) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix=f"{self.name}_in_"))
        for it in items:
            src = it.pdf if (use_pdf and it.pdf) else it.image
            ext = os.path.splitext(src)[1]
            shutil.copy(src, tmp / f"{it.key}{ext}")
        return tmp

    def _call(self, cmd: List[str], env: Optional[dict] = None) -> None:
        t0 = time.time()
        p = subprocess.run(cmd, env={**os.environ, **(env or {})}, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed ({p.returncode}):\n{p.stderr[-3000:]}")
        self.last_seconds = time.time() - t0


class ChandraEngine(_FolderCLI):
    name = "chandra"

    def __init__(self, method: str = "hf", max_output_tokens: Optional[int] = None, use_pdf: bool = False,
                 extra_env: Optional[dict] = None, keep_headers: bool = False):
        self.method, self.max_tokens, self.use_pdf, self.env = method, max_output_tokens, use_pdf, extra_env or {}
        self.keep_headers = keep_headers

    def run(self, items: List[PageItem]) -> Dict[str, str]:
        src = self._stage(items, self.use_pdf)
        dst = Path(tempfile.mkdtemp(prefix="chandra_out_"))
        cmd = ["chandra", str(src), str(dst), "--method", self.method]
        if self.max_tokens:
            cmd += ["--max-output-tokens", str(self.max_tokens)]
        if self.keep_headers:
            cmd += ["--include-headers-footers"]
        self._call(cmd, self.env)
        out = {}
        for it in items:
            md = dst / it.key / f"{it.key}.md"
            out[it.key] = md.read_text(encoding="utf-8") if md.exists() else ""
        shutil.rmtree(src, ignore_errors=True)
        return out


class SuryaEngine(_FolderCLI):
    name = "surya"

    def __init__(self, use_pdf: bool = False, extra_env: Optional[dict] = None):
        self.use_pdf, self.env = use_pdf, extra_env or {}

    def run(self, items: List[PageItem]) -> Dict[str, str]:
        src = self._stage(items, self.use_pdf)
        dst = Path(tempfile.mkdtemp(prefix="surya_out_"))
        self._call(["surya_ocr", str(src), "--output_dir", str(dst)], self.env)
        res_files = list(dst.rglob("results.json"))
        data = json.loads(res_files[0].read_text(encoding="utf-8")) if res_files else {}
        out = {}
        for it in items:
            pages = data.get(it.key) or data.get(f"{it.key}.png") or []
            out[it.key] = "\n\n".join(surya_page_to_markdown(p) for p in pages)
        shutil.rmtree(src, ignore_errors=True)
        return out


class CommandEngine(_FolderCLI):
    name = "command"

    def __init__(self, template: str, use_pdf: bool = False, html: bool = False):
        self.template, self.use_pdf, self.html = template, use_pdf, html

    def run(self, items: List[PageItem]) -> Dict[str, str]:
        src = self._stage(items, self.use_pdf)
        dst = Path(tempfile.mkdtemp(prefix="cmd_out_"))
        self._call(shlex.split(self.template.format(input=str(src), output=str(dst))))
        out = {}
        for it in items:
            hits = list(dst.rglob(f"{it.key}.md")) + list(dst.rglob(f"{it.key}.html"))
            txt = hits[0].read_text(encoding="utf-8") if hits else ""
            out[it.key] = html_to_markdown(txt) if (self.html or (hits and hits[0].suffix == ".html")) else txt
        return out


class MarkdownDirEngine(Engine):
    name = "mddir"

    def __init__(self, folder: str):
        self.folder = Path(folder)

    def run(self, items: List[PageItem]) -> Dict[str, str]:
        out = {}
        for it in items:
            p = self.folder / f"{it.key}.md"
            out[it.key] = p.read_text(encoding="utf-8") if p.exists() else ""
        return out


def make_engine(args) -> Engine:
    if args.engine == "folio":
        return FolioEngine(args.model, args.adapter, task=args.task, max_pixels=args.max_pixels, dtype=args.dtype,
                           load_4bit=args.load_4bit, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                           second_look=getattr(args, "second_look", False),
                           keep_headers=getattr(args, "keep_headers", False))
    if args.engine == "chandra":
        return ChandraEngine(method=args.chandra_method, max_output_tokens=args.max_new_tokens,
                             use_pdf=args.render == "native", keep_headers=getattr(args, "keep_headers", False))
    if args.engine == "surya":
        return SuryaEngine(use_pdf=args.render == "native")
    if args.engine == "command":
        return CommandEngine(args.command, use_pdf=args.render == "native")
    if args.engine == "mddir":
        return MarkdownDirEngine(args.md_dir)
    raise ValueError(args.engine)


def add_engine_args(ap) -> None:
    ap.add_argument("--engine", default="folio", choices=["folio", "chandra", "surya", "command", "mddir"])
    ap.add_argument("--model", default="infly/Infinity-Parser2-Flash")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--task", default="doc2json", choices=["doc2json", "doc2md"])
    ap.add_argument("--max-pixels", type=int, default=2_000_000,
                    help="visual budget (x1024 per token); Infinity-Parser2's own default is 16777216")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--second-look", action="store_true")
    ap.add_argument("--keep-headers", action="store_true",
                    help="keep page headers/footers in the Markdown (Resume-Bench does this by default; olmOCR-Bench "
                         "expects them dropped)")
    ap.add_argument("--chandra-method", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--command", default=None, help="for --engine command: e.g. 'mytool {input} {output}'")
    ap.add_argument("--md-dir", default=None, help="for --engine mddir")
