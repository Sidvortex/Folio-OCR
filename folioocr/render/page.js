// Injected into every synthetic page. Exposes window.__folio with the steps
// render.browser.Renderer calls in order: renderMath, wrapWords, trim, extract.
// Coordinates are CSS px relative to #page; Python scales them to image px.
(() => {
  const F = {};

  F.renderMath = () => {
    let failed = 0;
    document.querySelectorAll('[data-latex]').forEach((el) => {
      const tex = el.getAttribute('data-latex');
      const display = el.getAttribute('data-display') === '1';
      try {
        if (!window.katex) throw new Error('katex missing');
        window.katex.render(tex, el, { displayMode: display, throwOnError: true, output: 'html' });
      } catch (e) {
        el.textContent = tex;
        el.setAttribute('data-math-failed', '1');
        failed += 1;
      }
    });
    return failed;
  };

  // Split every text node inside a block into one span per word, so word and
  // line boxes can be measured. Math and SVG content are left alone.
  F.wrapWords = () => {
    document.querySelectorAll('#page [data-cat]').forEach((b) => {
      const cat = b.getAttribute('data-cat');
      if (cat === 'figure' || cat === 'formula') return;
      const walker = document.createTreeWalker(b, NodeFilter.SHOW_TEXT, {
        acceptNode(n) {
          const p = n.parentElement;
          if (!p || p.closest('[data-latex]') || p.closest('svg')) return NodeFilter.FILTER_REJECT;
          return /\S/.test(n.textContent) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
        },
      });
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      for (const n of nodes) {
        const frag = document.createDocumentFragment();
        for (const part of n.textContent.split(/([\s ]+)/)) {
          if (!part) continue;
          if (/^[\s ]+$/.test(part)) frag.appendChild(document.createTextNode(part));
          else {
            const s = document.createElement('span');
            s.className = '__w';
            s.textContent = part;
            frag.appendChild(s);
          }
        }
        n.parentNode.replaceChild(frag, n);
      }
    });
  };

  const visible = (el) => {
    const r = el.getBoundingClientRect();
    return r.width > 0 || r.height > 0;
  };

  // Remove every flow block that does not fit inside the page's text area
  // (whole blocks only: a label never describes a half-visible paragraph).
  F.trim = (limBottom, limRight) => {
    const page = document.getElementById('page');
    const pr = page.getBoundingClientRect();
    let removed = 0;
    for (let iter = 0; iter < 16; iter++) {
      const over = [...page.querySelectorAll('[data-cat]')].filter((el) => {
        if (el.closest('[data-fixed]') || !visible(el)) return false;
        const r = el.getBoundingClientRect();
        return r.bottom > pr.top + limBottom + 0.5 || r.right > pr.left + limRight + 0.5;
      });
      if (!over.length) break;
      // a caption never outlives its table/figure/formula: drop the whole group
      over.forEach((el) => { const g = el.closest('[data-group]'); (g || el).remove(); removed += 1; });
    }
    page.querySelectorAll('section').forEach((sec) => {
      const cats = [...sec.querySelectorAll('[data-cat]')];
      if (!cats.length || cats.every((c) => c.getAttribute('data-cat') === 'title')) sec.remove();
    });
    return removed;
  };

  // ------------------------------------------------------------ serialiser
  const BLOCK_DISPLAY = new Set(['block', 'flex', 'list-item', 'table', 'grid', 'table-row', 'table-cell', 'flow-root']);

  function wrapMark(t, m) {
    const s = t.replace(/[ \t ]+/g, ' ').trim();
    if (!s) return t;
    if (s.includes('\n')) return t;          // emphasis across a line break: keep plain
    const lead = /^[  ]/.test(t) ? ' ' : '';
    const trail = /[  ]$/.test(t) ? ' ' : '';
    return `${lead}${m}${s}${m}${trail}`;
  }

  function oneLine(s) {
    return s.split('\n').map((x) => x.replace(/[  ]+/g, ' ').trim()).filter(Boolean).join(' ');
  }

  function plainText(el) {
    let out = '';
    el.childNodes.forEach((n) => {
      if (n.nodeType === Node.TEXT_NODE) out += n.textContent;
      else if (n.nodeType === Node.ELEMENT_NODE) {
        if (n.tagName.toLowerCase() === 'svg' || n.getAttribute('aria-hidden') === 'true') return;
        if (n.hasAttribute('data-latex')) { out += `$${n.getAttribute('data-latex')}$`; return; }
        if (n.tagName === 'BR') { out += ' '; return; }
        out += plainText(n);
      }
    });
    return out.replace(/[\s ]+/g, ' ');
  }

  function tableHTML(t) {
    let out = '<table>';
    t.querySelectorAll('tr').forEach((tr) => {
      out += '<tr>';
      [...tr.children].forEach((c) => {
        if (c.tagName !== 'TD' && c.tagName !== 'TH') return;
        const tag = c.tagName.toLowerCase();
        const cs = c.colSpan > 1 ? ` colspan="${c.colSpan}"` : '';
        const rs = c.rowSpan > 1 ? ` rowspan="${c.rowSpan}"` : '';
        const txt = plainText(c).trim().replace(/</g, '&lt;').replace(/>/g, '&gt;');
        out += `<${tag}${cs}${rs}>${txt}</${tag}>`;
      });
      out += '</tr>';
    });
    return out + '</table>';
  }

  function ser(node) {
    if (node.nodeType === Node.TEXT_NODE) return node.textContent.replace(/[\s ]+/g, ' ');
    if (node.nodeType !== Node.ELEMENT_NODE) return '';
    const el = node;
    const tag = el.tagName.toUpperCase();
    if (tag === 'SVG' || tag === 'STYLE' || tag === 'SCRIPT' || el.getAttribute('aria-hidden') === 'true') return '';
    if (el.hasAttribute('data-latex')) {
      const tex = el.getAttribute('data-latex');
      return el.getAttribute('data-display') === '1' ? `\n$$${tex}$$\n` : `$${tex}$`;
    }
    if (tag === 'BR') return '\n';
    if (tag === 'STRONG' || tag === 'B') return wrapMark(serChildren(el, false), '**');
    if (tag === 'EM' || tag === 'I') return wrapMark(serChildren(el, false), '*');
    if (tag === 'UL' || tag === 'OL') {
      const items = [...el.children].filter((c) => c.tagName === 'LI');
      return '\n' + items.map((li, i) => (tag === 'OL' ? `${i + 1}. ` : '- ') + oneLine(serChildren(li, false))).join('\n') + '\n';
    }
    if (tag === 'TABLE') return '\n' + tableHTML(el) + '\n';
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return '';
    const flexRow = (cs.display === 'flex' || cs.display === 'inline-flex') && !cs.flexDirection.startsWith('column');
    const inner = serChildren(el, flexRow);
    return BLOCK_DISPLAY.has(cs.display) ? '\n' + inner + '\n' : inner;
  }

  function serChildren(el, flexRow) {
    const parts = [...el.childNodes].map(ser);
    if (flexRow) return parts.map((p) => p.replace(/\n+/g, '\n').trim()).filter(Boolean).join(' ');
    return parts.join('');
  }

  function blockText(el) {
    const cat = el.getAttribute('data-cat');
    if (cat === 'figure') return '';
    if (cat === 'formula') return `$$${el.getAttribute('data-latex')}$$`;
    if (cat === 'table' || el.tagName === 'TABLE') return tableHTML(el.tagName === 'TABLE' ? el : el.querySelector('table'));
    const cs = getComputedStyle(el);
    const flexRow = (cs.display === 'flex' || cs.display === 'inline-flex') && !cs.flexDirection.startsWith('column');
    let raw;
    if (el.tagName === 'UL' || el.tagName === 'OL') raw = ser(el);
    else raw = serChildren(el, flexRow);
    let text = raw.split('\n').map((x) => x.replace(/[ \t ]+/g, ' ').trim()).filter(Boolean).join('\n');
    if (cat === 'title') {
      const lvl = Math.max(1, Math.min(6, parseInt(el.getAttribute('data-level') || '2', 10)));
      text = '#'.repeat(lvl) + ' ' + oneLine(text).replace(/\*\*/g, '');
    }
    return text;
  }

  // ------------------------------------------------------------- geometry
  const rel = (r, pr) => [r.left - pr.left, r.top - pr.top, r.right - pr.left, r.bottom - pr.top];

  function union(boxes) {
    if (!boxes.length) return null;
    return boxes.reduce((a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])]);
  }

  function tightBox(el, pr) {
    const cat = el.getAttribute('data-cat');
    if (cat === 'figure' || cat === 'table' || el.tagName === 'TABLE') {
      const t = el.tagName === 'TABLE' ? el : (el.querySelector('table') || el);
      const target = cat === 'figure' ? (el.querySelector('svg') || el) : t;
      return rel(target.getBoundingClientRect(), pr);
    }
    if (cat === 'formula') {
      const bases = [...el.querySelectorAll('.katex-html .base')].map((b) => rel(b.getBoundingClientRect(), pr));
      return union(bases) || rel(el.getBoundingClientRect(), pr);
    }
    const parts = [];
    el.querySelectorAll('span.__w').forEach((w) => [...w.getClientRects()].forEach((r) => parts.push(rel(r, pr))));
    el.querySelectorAll('[data-latex]').forEach((m) => parts.push(rel(m.getBoundingClientRect(), pr)));
    el.querySelectorAll('svg').forEach((s) => parts.push(rel(s.getBoundingClientRect(), pr)));
    let box = union(parts.filter((b) => b[2] > b[0] && b[3] > b[1]));
    if (!box) return rel(el.getBoundingClientRect(), pr);
    if (el.tagName === 'UL' || el.tagName === 'OL') {   // include the bullet / number gutter
      const r = el.getBoundingClientRect();
      box[0] = Math.min(box[0], r.left - pr.left);
    }
    return box;
  }

  // A space goes between two words of a line when the DOM has whitespace
  // between them, or when they are visibly apart (flex items, table cells);
  // "Robot</strong>, CI/CD" stays "Robot, CI/CD".
  function spaceBetween(a, b, ba, bb) {
    const r = document.createRange();
    r.setStartAfter(a);
    r.setEndBefore(b);
    if (/[\s ]/.test(r.toString())) return true;
    const lh = Math.max(1, ba[3] - ba[1]);
    return bb[0] - ba[2] > 0.25 * lh;
  }

  function linesOf(el, bi, pr) {
    const words = [...el.querySelectorAll('span.__w')];
    const out = [];
    let cur = null;
    let broken = false;
    const maths = [...el.querySelectorAll('[data-latex]')].map((m) => rel(m.getBoundingClientRect(), pr));
    for (const w of words) {
      const rects = [...w.getClientRects()];
      if (!rects.length) continue;
      if (rects.length > 1) broken = true;
      const b = rel(rects[0], pr);
      const cy = (b[1] + b[3]) / 2;
      if (cur && cy >= cur.box[1] && cy <= cur.box[3] && b[0] >= cur.lastX - 2) {
        cur.text += (spaceBetween(cur.lastEl, w, cur.lastBox, b) ? ' ' : '') + w.textContent;
        cur.box = union([cur.box, b]);
        cur.lastX = b[2];
        cur.lastEl = w;
        cur.lastBox = b;
      } else {
        if (cur) out.push(cur);
        cur = { box: b, text: w.textContent, lastX: b[2], lastEl: w, lastBox: b };
      }
    }
    if (cur) out.push(cur);
    return out.map((l) => ({
      box: l.box,
      text: l.text,
      block: bi,
      math: maths.some((m) => m[3] > l.box[1] && m[1] < l.box[3]),
      broken,
    }));
  }

  const ORDER = { header: 0, page_footnote: 2, footer: 3 };

  F.extract = () => {
    const page = document.getElementById('page');
    const pr = page.getBoundingClientRect();
    const els = [...page.querySelectorAll('[data-cat]')].filter(visible);
    const items = els.map((el, i) => ({ el, i, rank: ORDER[el.getAttribute('data-cat')] ?? 1 }));
    items.sort((a, b) => a.rank - b.rank || a.i - b.i);
    const blocks = [];
    const lines = [];
    items.forEach(({ el }, bi) => {
      const cat = el.getAttribute('data-cat');
      blocks.push({ category: cat, box: tightBox(el, pr), text: blockText(el),
                    mathFailed: el.querySelector('[data-math-failed]') !== null || el.hasAttribute('data-math-failed') });
      if (cat !== 'figure' && cat !== 'formula' && cat !== 'table') lines.push(...linesOf(el, bi, pr));
    });
    return { width: pr.width, height: pr.height, blocks, lines };
  };

  window.__folio = F;
})();
