import react from '@vitejs/plugin-react';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';

/** Dev only: serve ./weights with HTTP Range support, so the app can load GPT-2 without the network. */
function localWeights(): Plugin {
  return {
    name: 'local-weights',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/local-weights', (req, res, next) => {
        const file = resolve('weights', (req.url ?? '').replace(/^\//, '').split('?')[0]);
        if (!file.startsWith(resolve('weights')) || !existsSync(file)) return next();
        const size = statSync(file).size;
        const m = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? '');
        res.setHeader('Accept-Ranges', 'bytes');
        if (!m) {
          res.setHeader('Content-Length', size);
          return createReadStream(file).pipe(res);
        }
        const start = +m[1];
        const end = m[2] ? +m[2] : size - 1;
        res.statusCode = 206;
        res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
        res.setHeader('Content-Length', end - start + 1);
        createReadStream(file, { start, end }).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), localWeights()],
  worker: { format: 'es' },
  build: { chunkSizeWarningLimit: 2000 },
});
