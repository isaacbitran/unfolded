// Tiny prose renderer: paragraphs, **bold**, *italic*, "- " lists, "1. " lists, $inline$ and $$display$$ KaTeX.
import katex from 'katex';

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function inline(s: string): string {
  const parts = s.split(/(\$[^$]+\$)/g);
  return parts
    .map((p) =>
      p.startsWith('$') && p.endsWith('$') && p.length > 2
        ? katex.renderToString(p.slice(1, -1), { throwOnError: false })
        : esc(p).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/\*([^*]+)\*/g, '<em>$1</em>'),
    )
    .join('');
}

export function renderProse(src: string): string {
  return src
    .trim()
    .split(/\n\s*\n/)
    .map((block) => {
      const b = block.trim();
      if (b.startsWith('$$') && b.endsWith('$$')) return `<div class="display">${katex.renderToString(b.slice(2, -2), { displayMode: true, throwOnError: false })}</div>`;
      const lines = b.split('\n');
      if (lines.every((l) => /^- /.test(l))) return `<ul>${lines.map((l) => `<li>${inline(l.slice(2))}</li>`).join('')}</ul>`;
      if (lines.every((l) => /^\d+\. /.test(l))) return `<ol>${lines.map((l) => `<li>${inline(l.replace(/^\d+\. /, ''))}</li>`).join('')}</ol>`;
      return `<p>${inline(b.replace(/\n/g, ' '))}</p>`;
    })
    .join('');
}

export const tex = (s: string, display = false) => katex.renderToString(s, { throwOnError: false, displayMode: display });
