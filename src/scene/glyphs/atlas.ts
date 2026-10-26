// One glyph atlas for every number in the scene, drawn with KaTeX Main (a Computer Modern clone)
// into a canvas once the font has loaded.

import * as THREE from 'three';
import katexMain from 'katex/dist/fonts/KaTeX_Main-Regular.ttf?url';

export const CHARS =
  '0123456789.,−-+×=()[]{}⋮⋯…∞%:/|<>^_*·#!?\'"' +
  'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ';

const CELL_W = 160;
const CELL_H = 160;
const FONT_PX = 100;
const COLS = 12;

export interface Glyph {
  /** u, v, w, h in texture space (v measured from the bottom) */
  uv: [number, number, number, number];
  /** advance width as a fraction of the cell height */
  advance: number;
}

export interface Atlas {
  texture: THREE.Texture;
  glyphs: Map<string, Glyph>;
  /** cell aspect (width / height) */
  aspect: number;
}

let atlasPromise: Promise<Atlas> | null = null;

export function loadAtlas(): Promise<Atlas> {
  if (!atlasPromise) atlasPromise = build();
  return atlasPromise;
}

async function build(): Promise<Atlas> {
  const face = new FontFace('UnfoldedMain', `url(${katexMain})`);
  try {
    await face.load();
    document.fonts.add(face);
  } catch {
    /* fall back to the browser serif */
  }
  const chars = Array.from(CHARS);
  const rows = Math.ceil(chars.length / COLS);
  const canvas = document.createElement('canvas');
  canvas.width = COLS * CELL_W;
  canvas.height = THREE.MathUtils.ceilPowerOfTwo(rows * CELL_H);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'white';
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'center';
  ctx.font = `${FONT_PX}px UnfoldedMain, 'Latin Modern Roman', 'Times New Roman', serif`;
  const glyphs = new Map<string, Glyph>();
  const baseline = CELL_H * 0.66;
  chars.forEach((ch, i) => {
    const cx = (i % COLS) * CELL_W;
    const cy = Math.floor(i / COLS) * CELL_H;
    // clip to the cell so wide glyphs (⋯, ∞, W) never bleed into a neighbour
    ctx.save();
    ctx.beginPath();
    ctx.rect(cx + 1, cy + 1, CELL_W - 2, CELL_H - 2);
    ctx.clip();
    ctx.fillText(ch, cx + CELL_W / 2, cy + baseline);
    ctx.restore();
    const adv = ctx.measureText(ch).width / CELL_H;
    glyphs.set(ch, {
      uv: [cx / canvas.width, 1 - (cy + CELL_H) / canvas.height, CELL_W / canvas.width, CELL_H / canvas.height],
      advance: ch === ' ' ? 0.28 : adv,
    });
  });
  const texture = new THREE.CanvasTexture(canvas);
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.anisotropy = 8;
  texture.needsUpdate = true;
  return { texture, glyphs, aspect: CELL_W / CELL_H };
}
