// Capture frames along the story path for the README GIF.
// node scripts/frames.mjs <from> <to> <step> [--w 960 --h 540 --url …] → shots/frames/0000.png …
import { chromium } from '@playwright/test';
import { mkdirSync, rmSync } from 'node:fs';
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : d; };
const url = opt('--url', 'http://localhost:5391');
const w = +opt('--w', 960), h = +opt('--h', 540);
const dir = opt('--out', 'shots/frames');
const [from, to, step] = args.map(Number);
rmSync(dir, { recursive: true, force: true });
mkdirSync(dir, { recursive: true });
const b = await chromium.launch({ args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist'] });
const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
await p.goto(`${url}/?s=${from}&clean`);
await p.waitForFunction(() => (window.__perf?.frames ?? 0) > 40);
let i = 0;
for (let s = from; s <= to + 1e-9; s += step) {
  await p.evaluate(async (s) => {
    const { story } = await import('/src/state/store.ts');
    story.target = story.s = s;
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  }, s);
  await p.screenshot({ path: `${dir}/${String(i++).padStart(4, '0')}.png` });
}
console.log(`${i} frames → ${dir}`);
await b.close();
