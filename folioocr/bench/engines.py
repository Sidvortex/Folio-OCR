"""OCR engines behind one interface: page images in, Markdown out.

    folio     our model (or plain Infinity-Parser2-Flash / any Qwen3.5-family
              checkpoint), run in-process with transformers
    chandra   Datalab's ``chandra`` CLI (pip install chandra-ocr[hf])
    surya     Datalab's ``surya_ocr`` CLI (pip install surya-ocr; spawns its own
              vLLM / llama.cpp server)
    command   any tool: a shell template with {input} and {output} folders that
              leaves <stem>.md files in {output}
    mddir     precomputed Markdown files (<key>.md), e.g. from a paid API
    openai    an OpenAI-compatible endpoint (a vLLM server, or an API whose
              terms allow your use of its output)

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
from dataclasses import dataclass
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

    def run_full(self, items: List[PageItem]) -> Dict[str, dict]:
        """{key: {"markdown", "blocks" (layout, or None), "raw"}}: what teacher labelling keeps."""
        return {k: {"markdown": md, "blocks": None, "raw": md} for k, md in self.run(items).items()}

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

    def _parse(self, items: List[PageItem]) -> Dict[str, "object"]:
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
                out[it.key] = r
                self.stats.append({"key": it.key, "tokens": r.tokens, "seconds": r.seconds, "looped": r.looped,
                                   "retried": r.retried, "parse_ok": r.parse_ok})
        return out

    def run(self, items: List[PageItem]) -> Dict[str, str]:
        return {k: r.markdown for k, r in self._parse(items).items()}

    def run_full(self, items: List[PageItem]) -> Dict[str, dict]:
        out = {}
        for k, r in self._parse(items).items():
            ok = r.task == "doc2json" and r.parse_ok and not r.looped and not r.truncated
            out[k] = {"markdown": r.markdown, "blocks": [b.to_dict() for b in r.blocks] if ok else None,
                      "raw": r.raw, "looped": r.looped, "truncated": r.truncated}
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


class OpenAIEngine(Engine):
    """Any OpenAI-compatible chat endpoint with image input: a vLLM server
    (``vllm serve infly/Infinity-Parser2-Pro``, ``vllm serve allenai/olmOCR-2-7B-1025``)
    or a hosted API whose terms allow what you do with the output.

    ``prompt``: ``doc2json`` / ``doc2md`` (Infinity-Parser2's prompts, thinking
    off), ``olmocr`` (olmOCR's v4 prompt; its YAML front matter is stripped), or
    any other text sent as is.
    """
    name = "openai"

    def __init__(self, base_url: str, model: str, prompt: str = "doc2md", max_tokens: int = 8192,
                 api_key_env: str = "OPENAI_API_KEY", longest_side: Optional[int] = None,
                 max_pixels: Optional[int] = None, timeout: float = 900.0, keep_headers: bool = False):
        self.base_url, self.model, self.prompt_name = base_url.rstrip("/"), model, prompt
        self.max_tokens, self.timeout = max_tokens, timeout
        self.api_key = os.environ.get(api_key_env, "EMPTY")
        self.longest_side, self.max_pixels = longest_side, max_pixels
        self.keep_headers = keep_headers
        if prompt in ("doc2json", "doc2md"):
            from .. import prompts
            self.text = prompts.prompt_for(prompt)
        elif prompt == "olmocr":
            from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
            self.text = build_no_anchoring_v4_yaml_prompt()
            self.longest_side = self.longest_side or 1288
        else:
            self.text = prompt

    def _image_url(self, path: str) -> str:
        import base64
        import io
        from ..imaging import fit_image
        img = Image.open(path).convert("RGB")
        if self.longest_side and max(img.size) != self.longest_side:
            sc = self.longest_side / max(img.size)
            img = img.resize((max(1, round(img.width * sc)), max(1, round(img.height * sc))), Image.LANCZOS)
        if self.max_pixels:
            img = fit_image(img, self.max_pixels)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _ask(self, path: str) -> str:
        import requests
        body = {"model": self.model, "temperature": 0.0, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": self._image_url(path)}},
                                                          {"type": "text", "text": self.text}]}]}
        if self.prompt_name in ("doc2json", "doc2md"):
            body["chat_template_kwargs"] = {"enable_thinking": False}     # vLLM; ignored elsewhere
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    def _convert(self, raw: str) -> dict:
        from ..model.generate import strip_md_fences
        from ..schema import parse_output, to_markdown
        if self.prompt_name == "doc2json":
            pr = parse_output(raw)
            return {"markdown": to_markdown(pr.blocks, keep_non_body=self.keep_headers),
                    "blocks": [b.to_dict() for b in pr.blocks] if pr.ok else None, "raw": raw}
        text = raw
        if self.prompt_name == "olmocr":
            from ..data.sources.common import normalize_markdown
            text = normalize_markdown(raw)
        return {"markdown": strip_md_fences(text), "blocks": None, "raw": raw}

    def run_full(self, items: List[PageItem]) -> Dict[str, dict]:
        out = {}
        for it in items:
            try:
                out[it.key] = self._convert(self._ask(it.image))
            except Exception as e:  # one failed request must not end a long run
                out[it.key] = {"markdown": "", "blocks": None, "raw": "", "error": str(e)[:500]}
        return out

    def run(self, items: List[PageItem]) -> Dict[str, str]:
        return {k: v["markdown"] for k, v in self.run_full(items).items()}


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
    if args.engine == "openai":
        if not (args.api_base and args.api_model):
            raise SystemExit("--engine openai needs --api-base and --api-model")
        return OpenAIEngine(args.api_base, args.api_model, prompt=args.api_prompt, max_tokens=args.max_new_tokens,
                            longest_side=args.api_longest_side, max_pixels=args.max_pixels,
                            keep_headers=getattr(args, "keep_headers", False))
    raise ValueError(args.engine)


def add_engine_args(ap) -> None:
    ap.add_argument("--engine", default="folio", choices=["folio", "chandra", "surya", "command", "mddir", "openai"])
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
    ap.add_argument("--api-base", default=None, help="for --engine openai: e.g. http://localhost:8000/v1")
    ap.add_argument("--api-model", default=None, help="for --engine openai: the served model name")
    ap.add_argument("--api-prompt", default="doc2md", help="for --engine openai: doc2json, doc2md, olmocr or a prompt")
    ap.add_argument("--api-longest-side", type=int, default=None, help="for --engine openai: resize pages first")
