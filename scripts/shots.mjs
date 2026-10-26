// Render walkthrough positions to PNGs: node scripts/shots.mjs 2.1 2.5 2.95 [--url http://localhost:5173] [--w 1600 --h 900]
import { chromium } from '@playwright/test';
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : d; };
const url = opt('--url', 'http://localhost:5173');
const w = +opt('--w', 1600), h = +opt('--h', 900), dpr = +opt('--dpr', 1);
const extra = opt('--q', '');
const browser = await chromium.launch({ args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: dpr });
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('[console]', m.type(), m.text().slice(0, 300)); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
for (const s of args) {
  await page.goto(`${url}/?s=${s}${extra ? '&' + extra : ''}`);
  await page.waitForFunction(() => (window.__perf?.frames ?? 0) > 40, null, { timeout: 60000 });
  await page.waitForTimeout(400);
  const ms = await page.evaluate(() => window.__perf.ms);
  const file = `shots/s${s}.png`;
  await page.screenshot({ path: file });
  console.log(file, `frame ${ms.toFixed(1)} ms`);
}
await browser.close();
