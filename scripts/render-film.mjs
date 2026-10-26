// Render the demo film frame by frame (deterministic seek, no real-time dependence) and pipe the
// frames straight into ffmpeg. node scripts/render-film.mjs out.mp4 [--fps 60] [--dpr 2] [--from 0 --to 112] [--stills 1,10,20]
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : d; };
const url = opt('--url', 'http://localhost:5391');
const fps = +opt('--fps', 60), dpr = +opt('--dpr', 2);
const from = +opt('--from', 0);
const stills = opt('--stills', '');
const ffmpeg = opt('--ffmpeg', 'ffmpeg');
const out = args[0] ?? 'docs/unfolded-demo.mp4';
const b = await chromium.launch({ args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist'] });
const p = await b.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: dpr });
p.on('pageerror', (e) => console.log('[pageerror]', e.message));
await p.goto(`${url}/?video`);
await p.waitForFunction(() => typeof window.__videoSeek === 'function' && (window.__perf?.frames ?? 0) > 30, null, { timeout: 60000 });
const total = await p.evaluate(() => window.__filmSeconds);
const to = +opt('--to', total);
if (stills) {
  for (const t of stills.split(',').map(Number)) {
    await p.evaluate((t) => window.__videoSeek(t), t);
    await p.evaluate(() => window.__videoSeek(arguments[0]), t).catch(() => {});
    await p.screenshot({ path: `shots/film-${t}.png` });
    console.log(`shots/film-${t}.png`);
  }
  await b.close();
  process.exit(0);
}
const enc = spawn(ffmpeg, ['-y', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
  '-vf', 'scale=1920:1080:flags=lanczos,format=yuv420p', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16',
  '-profile:v', 'high', '-movflags', '+faststart', out], { stdio: ['pipe', 'ignore', 'inherit'] });
const n = Math.round((to - from) * fps);
const t0 = Date.now();
for (let i = 0; i < n; i++) {
  const t = from + i / fps;
  await p.evaluate((t) => window.__videoSeek(t), t);
  const jpg = await p.screenshot({ type: 'jpeg', quality: 93 });
  if (!enc.stdin.write(jpg)) await new Promise((r) => enc.stdin.once('drain', r));
  if (i % 300 === 0) console.log(`frame ${i}/${n}  t=${t.toFixed(1)}s  ${((Date.now() - t0) / 1000).toFixed(0)}s elapsed`);
}
enc.stdin.end();
await new Promise((r) => enc.on('close', r));
console.log(`${n} frames → ${out} in ${((Date.now() - t0) / 60000).toFixed(1)} min`);
await b.close();
