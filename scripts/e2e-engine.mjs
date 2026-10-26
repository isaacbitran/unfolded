// End-to-end engine check in a real browser: load weights, run a prompt, generate tokens.
// node scripts/e2e-engine.mjs [--cpu]
import { chromium } from '@playwright/test';
const cpu = process.argv.includes('--cpu');
const b = await chromium.launch({ args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU'] });
const p = await b.newPage({ viewport: { width: 1400, height: 850 } });
p.on('pageerror', (e) => console.log('[pageerror]', e.message));
p.on('console', (m) => m.type() === 'error' && !m.text().includes('unmount') && console.log('[console]', m.text().slice(0, 200)));
await p.goto(`http://localhost:5391/${cpu ? '?cpu' : ''}`);
await p.waitForSelector('.promptbar input');
console.log('webgpu in page:', await p.evaluate(() => !!navigator.gpu));
const t0 = Date.now();
await p.fill('.promptbar input', 'The quick brown fox jumps over the lazy');
await p.click('.promptbar button');
await p.waitForFunction(() => document.querySelector('.status')?.textContent?.startsWith('live'), null, { timeout: 300000 });
console.log('status:', await p.textContent('.status'), `(${((Date.now() - t0) / 1000).toFixed(1)} s incl. load)`);
const info = await p.evaluate(async () => {
  const { useApp } = await import('/src/state/store.ts');
  const t = useApp.getState().trace;
  return { prompt: t.prompt, T: t.T, demo: t.demo, top: t.positions[t.T - 1].top.slice(0, 3).map((c) => `${c.text}:${c.p.toFixed(3)}`) };
});
console.log(info);
const gen = await p.evaluate(async () => {
  const { generate } = await import('/src/engine/client.ts');
  const { useApp } = await import('/src/state/store.ts');
  useApp.getState().setSampling({ temperature: 0, topK: 0, topP: 1 });
  const t0 = performance.now();
  await generate(5, 1);
  const t = useApp.getState().trace;
  return { prompt: t.prompt, T: t.T, ms: performance.now() - t0 };
});
console.log('greedy +5:', gen);
await b.close();
