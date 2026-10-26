import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import 'katex/dist/katex.min.css';
import './index.css';
import App from './App';
import { loadAtlas } from './scene/glyphs/atlas';

// ?clean hides all UI (for recordings)
const q = new URLSearchParams(location.search);
if (q.has('clean') || q.has('video')) document.body.classList.add('clean');
if (q.has('video')) document.body.classList.add('video');

// Fonts and the glyph atlas are ready before the first frame, so nothing re-lays out.
loadAtlas().finally(() =>
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  ),
);
