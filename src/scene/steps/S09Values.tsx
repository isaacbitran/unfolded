// Step 9 — weighted sum of values. The focused head's value vectors (green, 64 entries each) stand
// above their tokens with the last token's attention row beneath them. One token at a time, a
// ghost copy of v_j flies into z_T with brightness ∝ A_{T,j}, together with a stream of particles
// whose count ∝ A_{T,j}; z_T's entries roll through the running sums Σ_{j'≤j} A_{T,j'} v_{j'} and
// land on the real z (layers[0].z, head slice).

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { GlyphField, layoutText, useAtlas, type GlyphFieldHandle, type GlyphItem } from '../glyphs/GlyphField';
import { Columns, type ColumnsHandle } from '../primitives/Columns';
import { Lines, bracketPoints } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TokenHeader, type TokenHeaderHandle } from '../primitives/TokenHeader';
import { C, L, colX, easeInOut, env, fmt, seg } from '../theme';
import { useStep } from '../useStep';
import { columnBox, glow, rms, visibleEntries } from '../vectors';
import { COL_TOP, TOKEN_ID_Y, TOKEN_LABEL_Y } from './S02Embedding';
import { HeatCells, focusHead, heat, rollGlyphs, type HeatCellsHandle, type RollState } from './S08Softmax';

const D = 768;
const DH = 64;

/** Deterministic PRNG so particle layouts are identical on every render. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function S09Values({ index, trace }: { index: number; trace: SceneTrace }) {
  const atlas = useAtlas();
  const T = trace.T;
  const last = T - 1;
  const h = useMemo(() => focusHead(trace), [trace]);
  const off = DH * h;
  const L0 = trace.layers[0];
  const Arow = useMemo(() => Array.from(L0.pattern.subarray(h * T * T + last * T, h * T * T + T * T)), [L0, h, T, last]);
  const top = useMemo(() => Arow.reduce((b, a, j) => (a > Arow[b] ? j : b), 0), [Arow]);

  const xs = useMemo(() => trace.tokens.map((_, k) => colX(k, T)), [trace, T]);
  const zx = xs[last] + 2.9;
  const box = columnBox(0, COL_TOP);
  const wY = box.y0 - 0.75; // weights row
  const wS = 0.92; // weight square size

  // flow windows: one token after another, left to right
  const span = 0.36;
  const win = (j: number): [number, number] => {
    const a = 0.26 + (T > 1 ? (span * j) / (T - 1) : 0);
    return [a, a + 0.14];
  };

  const valEntries = useMemo(() => trace.tokens.map((_, j) => visibleEntries(L0.v, j * D + off, DH)), [trace, L0, off]);
  const zEntries = useMemo(() => visibleEntries(L0.z, last * D + off, DH), [L0, last, off]);

  // glyphs that fly with a z-arc: ghost value copies and particles
  const flights = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const out: GlyphItem[] = [];
    const green = new THREE.Color(C.value);
    // ghost copies of each value column flying into z (brightness ∝ weight)
    trace.tokens.forEach((_, j) => {
      const [a, b] = win(j);
      const w = Arow[j];
      const k = 0.25 + 1.6 * w;
      valEntries[j].forEach((e, r) => {
        if (e.idx < 0) return;
        const y = COL_TOP - r * L.rowH;
        out.push(
          ...layoutText(e.text, zx + 0.36, y, 0.05, {
            h: L.glyphH,
            color: green,
            k,
            alpha: Math.min(1, 0.12 + 2 * w),
            align: 'right',
            from: [xs[j] + 0.36, y, 0.05],
            delay: [a + r * 0.004, b - 0.02 + r * 0.004],
            appear: a,
            vanish: b + 0.01 + r * 0.004,
          }),
        );
      });
    });
    // particles: count ∝ weight, each with its own path and timing inside the token's window
    const rnd = mulberry32(1234 + h);
    const zb = columnBox(zx, COL_TOP);
    trace.tokens.forEach((_, j) => {
      const [a, b] = win(j);
      const n = Math.round(Arow[j] * 140);
      const vb = columnBox(xs[j], COL_TOP);
      for (let q = 0; q < n; q++) {
        const t0 = a + rnd() * (b - a - 0.07);
        const fx = vb.x0 + 0.15 + rnd() * (vb.x1 - vb.x0 - 0.3);
        const fy = vb.y0 + 0.15 + rnd() * (vb.y1 - vb.y0 - 0.3);
        const tx = zb.x0 + 0.2 + rnd() * (zb.x1 - zb.x0 - 0.4);
        const ty = zb.y0 + 0.2 + rnd() * (zb.y1 - zb.y0 - 0.4);
        out.push({
          ch: '·', x: tx, y: ty, z: 0.1, h: 0.95, color: green, k: 1.6, alpha: 0.95,
          from: [fx, fy, 0.1], delay: [t0, t0 + 0.06], appear: t0 - 0.01, vanish: t0 + 0.065,
        });
      }
    });
    return out;
  }, [atlas, trace, Arow, valEntries, xs, zx, h]); // eslint-disable-line react-hooks/exhaustive-deps

  // glyphs that stay in their plane: z_T's rolling entries and the weight numbers
  const items = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const out: GlyphItem[] = [];
    const green = new THREE.Color(C.value);
    const white = new THREE.Color(C.token);
    const zScale = rms(L0.z, last * D + off, DH);
    // z entries: zeros, then the running sum after each token's contribution, ending on the real z
    zEntries.forEach((e, r) => {
      const y = COL_TOP - r * L.rowH;
      if (e.idx < 0) {
        out.push(...layoutText('⋮', zx, y, 0, { h: L.glyphH, color: green, k: 0.8, appear: win(0)[0] - 0.04 }));
        return;
      }
      const st: RollState[] = [{ text: fmt(0), t: win(0)[0] - 0.04, k: 0.45 }];
      let acc = 0;
      trace.tokens.forEach((_, j) => {
        acc += Arow[j] * L0.v[j * D + off + e.idx];
        const v = j === last ? e.value : acc;
        st.push({ text: fmt(v), t: win(j)[1] - 0.01, k: glow(v, zScale) * 1.05 });
      });
      out.push(...rollGlyphs(st, zx + 0.36, y, 0, { h: L.glyphH, color: green, align: 'right' }));
    });
    // weight numbers inside the squares
    Arow.forEach((a, j) => out.push(...layoutText(fmt(a), xs[j], wY, 0.02, { h: 0.26, color: white, k: 0.7 + 0.5 * Math.sqrt(a), appear: 0.1 })));
    return out;
  }, [atlas, trace, L0, Arow, valEntries, zEntries, xs, zx, off, h, last, wY]); // eslint-disable-line react-hooks/exhaustive-deps

  const centers = useMemo(() => xs.map((x) => [x, wY, -0.02] as [number, number, number]), [xs, wY]);
  const zBracket = useMemo(() => {
    const b = columnBox(zx, COL_TOP);
    return bracketPoints(b.x0, b.x1, b.y0, b.y1, 0);
  }, [zx]);
  const topBox = useMemo(() => {
    const x = xs[top];
    const s = wS / 2 + 0.06;
    return [x - s, wY + s, 0, x + s, wY + s, 0, x + s, wY + s, 0, x + s, wY - s, 0, x + s, wY - s, 0, x - s, wY - s, 0, x - s, wY - s, 0, x - s, wY + s, 0];
  }, [xs, top, wY]);

  const vals = useRef<ColumnsHandle>(null);
  const header = useRef<TokenHeaderHandle>(null);
  const glyphs = useRef<GlyphFieldHandle>(null);
  const flyRef = useRef<GlyphFieldHandle>(null);
  const cells = useRef<HeatCellsHandle>(null);
  const zBr = useRef<THREE.LineSegments>(null);
  const topLine = useRef<THREE.LineSegments>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);

  useStep(index, (p, v) => {
    // sharpened crossfade: this step's objects arrive late and leave early, so neighbours barely overlap
    const e = seg(v, 0.6, 1);
    const vis = e * e * (3 - 2 * e);
    const on = vis > 0.001;
    const inA = vis * easeInOut(seg(p, 0.02, 0.12));
    header.current?.setOpacity(vis);
    header.current?.setIdOpacity(0);
    header.current?.setFocus(last, seg(p, 0.12, 0.22));
    vals.current?.setOpacity(inA);
    glyphs.current?.setProgress(p);
    glyphs.current?.setOpacity(on ? vis : 0);
    flyRef.current?.setProgress(p);
    flyRef.current?.setOpacity(on ? vis : 0);
    flyRef.current?.setArc(2.2);
    if (zBr.current) {
      (zBr.current.material as THREE.LineBasicMaterial).opacity = 0.85 * vis * seg(p, 0.2, 0.28);
      zBr.current.visible = on;
    }
    const cl = cells.current;
    if (cl) {
      const rowIn = vis * easeInOut(seg(p, 0.08, 0.16));
      Arow.forEach((a, j) => {
        const [w0, w1] = win(j);
        const pulse = 1 + 1.3 * env(p, w0, w1, 0.04);
        const [r, g, b] = heat(a);
        cl.set(j, r * pulse * rowIn, g * pulse * rowIn, b * pulse * rowIn);
      });
      cl.commit(on);
    }
    const share = vis * seg(p, 0.79, 0.86);
    if (topLine.current) {
      (topLine.current.material as THREE.LineBasicMaterial).opacity = 0.9 * share;
      topLine.current.visible = on && share > 0.001;
    }
    fade(labels.current[0], vis * seg(p, 0.06, 0.14)); // v_j shape
    fade(labels.current[1], vis * seg(p, 0.1, 0.18)); // A row
    fade(labels.current[2], vis * seg(p, 0.2, 0.28)); // z_T name
    fade(labels.current[3], vis * seg(p, 0.86, 0.94)); // z_T shape
    fade(labels.current[4], vis * seg(p, 0.16, 0.26)); // formula
    fade(labels.current[5], share); // top share
  });

  const pct = Math.round(Arow[top] * 100);
  const topTok = trace.tokens[top].text.replace(/^ /, '');
  return (
    <group>
      <TokenHeader ref={header} trace={trace} y={TOKEN_LABEL_Y} idY={TOKEN_ID_Y} />
      <Columns ref={vals} data={L0.v} D={D} xs={xs} y0={COL_TOP} color={C.value} offset={off} len={DH} />
      <HeatCells ref={cells} centers={centers} size={wS} gap={0.04} />
      <GlyphField ref={glyphs} items={items} />
      <GlyphField ref={flyRef} items={flights} />
      <Lines ref={zBr} points={zBracket} color={C.value} opacity={0} />
      <Lines ref={topLine} points={topBox} color={C.attn} opacity={0} />
      <Label ref={(e) => void (labels.current[0] = e)} position={[xs[0] - 0.95, COL_TOP - 1.2, 0]} tex={`v_j \\in \\mathbb{R}^{64}`} size={15} color={C.value} align="right" />
      <Label ref={(e) => void (labels.current[1] = e)} position={[xs[0] - 0.95, wY, 0]} tex={`A_{T,j}`} size={17} color={C.attn} align="right" />
      <Label ref={(e) => void (labels.current[2] = e)} position={[zx, TOKEN_LABEL_Y, 0]} tex={`z_T`} size={26} color={C.value} />
      <Label ref={(e) => void (labels.current[3] = e)} position={[zx, box.y0 - 0.3, 0]} tex={`\\mathbb{R}^{64}`} size={13} color="#9a9ea6" />
      <Label ref={(e) => void (labels.current[4] = e)} position={[(xs[0] + zx) / 2, wY - 1.45, 0]} tex={`z_T = \\sum_{j\\le T} A_{T,j}\\, v_j \\qquad \\text{head } ${h + 1}`} size={20} />
      <Label ref={(e) => void (labels.current[5] = e)} position={[xs[top], wY - wS / 2 - 0.3, 0.1]} text={`${pct}% from “${topTok}”`} size={14} color={C.attn} />
    </group>
  );
}
