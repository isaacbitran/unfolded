// Render the link-preview image (1200×630) from the film's title frame. node scripts/og.mjs
import { chromium } from '@playwright/test';
const b = await chromium.launch({ args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist'] });
const p = await b.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 2 });
await p.goto(`${process.argv[2] ?? 'http://localhost:5391'}/?video`);
await p.waitForFunction(() => typeof window.__videoSeek === 'function' && (window.__perf?.frames ?? 0) > 30);
await p.evaluate(() => window.__videoSeek(1.6));
await p.screenshot({ path: 'public/og.png' });
await b.close();
console.log('public/og.png');
