// Dev helper: list 3D DOM labels (text, opacity, screen rect) at a story position. node scripts/probe.mjs 2.3
import { chromium } from '@playwright/test';
const s = process.argv[2] ?? '2.3';
const b = await chromium.launch({ args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist'] });
const p = await b.newPage({ viewport: { width: 1600, height: 900 } });
await p.goto(`http://localhost:5391/?s=${s}`);
await p.waitForFunction(() => (window.__perf?.frames ?? 0) > 40);
console.log(await p.evaluate(() => [...document.querySelectorAll('.label3d')].map((e) => {
  const r = e.getBoundingClientRect();
  return `${e.textContent.slice(0, 24).padEnd(24)} op=${e.style.opacity} vis=${e.style.visibility} @${Math.round(r.x)},${Math.round(r.y)} ${Math.round(r.width)}x${Math.round(r.height)}`;
}).join('\n')));
await b.close();
