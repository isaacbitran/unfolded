// Step 8 — scale, mask, softmax. The T×T score grid of the focused head of layer 0:
// every cell divides by √64 = 8 (numbers roll to the new value in a diagonal wave), the upper
// triangle (j > i) becomes −∞ under a sliding glass sheet, and then each row exponentiates,
// is divided by its total, and turns into heat squares whose values are the real pattern A_h.
//
// This file also exports the helpers shared by steps 9 and 10: `focusHead`, `HeatCells`
// (an instanced quad grid with per-cell colors set every frame) and `rollGlyphs`.

import { forwardRef, useImperativeHandle, useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { GlyphField, layoutText, useAtlas, type GlyphFieldHandle, type GlyphItem, type TextOpts } from '../glyphs/GlyphField';
import { Lines } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TextPlane, measureText, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { C, easeInOut, env, fmt, seg } from '../theme';
import { useStep } from '../useStep';

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

/**
 * The head of layer 0 that steps 7–9 focus on: the head whose last query row, restricted to
 * positions 1..T−1 and renormalized to sum to 1, has the highest Shannon entropy. Position 0 is
 * excluded because most heads dump spare attention there (the "attention sink"); what is left
 * shows how broadly the head reads the rest of the sentence. Ties go to the lowest h.
 */
export function focusHead(trace: SceneTrace): number {
  const T = trace.T;
  const A = trace.layers[0].pattern;
  if (T < 3) return 0;
  let best = 0;
  let bestH = -Infinity;
  for (let h = 0; h < 12; h++) {
    const o = h * T * T + (T - 1) * T;
    let s = 0;
    for (let j = 1; j < T; j++) s += A[o + j];
    let H = 0;
    if (s > 0) {
      for (let j = 1; j < T; j++) {
        const a = A[o + j] / s;
        if (a > 0) H -= a * Math.log(a);
      }
    }
    if (H > bestH + 1e-9) {
      bestH = H;
      best = h;
    }
  }
  return best;
}

/** Number format for grid cells: like `fmt`, but huge values switch to compact exponent form. */
export function fmtCell(v: number): string {
  if (!Number.isFinite(v)) return fmt(v);
  if (Math.abs(v) >= 1e4) return v.toExponential(1).replace('e+', 'e').replace('-', '−');
  return fmt(v);
}

export interface RollState {
  text: string;
  /** progress at which this state rolls in (≤ 0 means present from the start) */
  t: number;
  k?: number;
  color?: THREE.Color;
}

/**
 * Glyphs for one number slot that changes value over the step: each new state rolls up into place
 * from slightly below while the previous one fades out. Pure function of the glyph field's progress.
 */
export function rollGlyphs(
  states: RollState[],
  x: number,
  y: number,
  z: number,
  o: Omit<TextOpts, 'from' | 'delay' | 'appear' | 'vanish' | 'k' | 'color'> & { color: THREE.Color; dur?: number; drop?: number; vanishAt?: number },
): GlyphItem[] {
  const out: GlyphItem[] = [];
  const dur = o.dur ?? 0.045;
  const drop = o.drop ?? o.h * 0.6;
  states.forEach((s, n) => {
    const next = states[n + 1];
    const first = s.t <= 0;
    out.push(
      ...layoutText(s.text, x, y, z, {
        ...o,
        color: s.color ?? o.color,
        k: s.k ?? 1,
        from: first ? undefined : [x, y - drop, z],
        delay: first ? undefined : [s.t, s.t + dur],
        appear: first ? -1 : s.t,
        vanish: next ? next.t + 0.012 : (o.vanishAt ?? 3),
      }),
    );
  });
  return out;
}

export interface HeatCellsHandle {
  /** set cell n's color (linear RGB, additive on black) */
  set(n: number, r: number, g: number, b: number): void;
  /** upload colors after a batch of `set` calls; hides the mesh when `on` is false */
  commit(on: boolean): void;
}

/** Instanced square cells (one draw call). Additive blending: black means invisible. */
export const HeatCells = forwardRef<HeatCellsHandle, { centers: [number, number, number][]; size: number; gap?: number }>(function HeatCells(
  { centers, size, gap = 0.07 },
  ref,
) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const geo = useMemo(() => new THREE.PlaneGeometry(1, 1), []);
  const mat = useMemo(
    () => new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }),
    [],
  );
  const color = useMemo(() => new THREE.Color(), []);
  useLayoutEffect(() => {
    const m = mesh.current;
    if (!m) return;
    const o = new THREE.Object3D();
    centers.forEach((c, n) => {
      o.position.set(c[0], c[1], c[2]);
      o.scale.setScalar(size * (1 - gap));
      o.updateMatrix();
      m.setMatrixAt(n, o.matrix);
      m.setColorAt(n, color.setRGB(0, 0, 0));
    });
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
    m.count = centers.length;
  }, [centers, size, gap, color]);
  useImperativeHandle(ref, () => ({
    set(n, r, g, b) {
      mesh.current?.setColorAt(n, color.setRGB(r, g, b));
    },
    commit(on) {
      const m = mesh.current;
      if (!m) return;
      m.visible = on;
      if (m.instanceColor) m.instanceColor.needsUpdate = true;
    },
  }));
  return <instancedMesh ref={mesh} args={[geo, mat, Math.max(1, centers.length)]} frustumCulled={false} renderOrder={3} />;
});

/** sRGB hex → linear RGB triple (instance colors are linear). */
export const lin = (hex: string): [number, number, number] => {
  const c = new THREE.Color(hex);
  return [c.r, c.g, c.b];
};

/** Heat color for an attention weight a ∈ [0,1]: dark → purple, sqrt so small weights still read. */
export function heat(a: number, max = 0.3): [number, number, number] {
  const [r, g, b] = PURPLE;
  const k = 0.01 + max * Math.sqrt(Math.max(0, a));
  return [r * k, g * k, b * k];
}
const PURPLE = lin(C.attn);
const WARM = lin(C.token);

// ---------------------------------------------------------------------------------------------
// Step 8
// ---------------------------------------------------------------------------------------------

/** Timing (step progress) of each phase; rows/cells get staggered windows inside them. */
const PH = { scale: [0.08, 0.27], mask: [0.32, 0.5], soft: [0.55, 0.9] } as const;

export function S08Softmax({ index, trace }: { index: number; trace: SceneTrace }) {
  const atlas = useAtlas();
  const T = trace.T;
  const h = useMemo(() => focusHead(trace), [trace]);
  const S = useMemo(() => trace.layers[0].scores.subarray(h * T * T, (h + 1) * T * T), [trace, h, T]);
  const A = useMemo(() => trace.layers[0].pattern.subarray(h * T * T, (h + 1) * T * T), [trace, h, T]);

  // geometry: square grid centered at the origin
  const c = Math.min(1.25, 9.5 / T);
  const W = c * T;
  const x0 = -W / 2;
  const y0 = W / 2;
  const cx = (j: number) => x0 + c * (j + 0.5);
  const cy = (i: number) => y0 - c * (i + 0.5);
  const gh = Math.min(0.3, c * 0.24);
  const tokH = Math.min(0.32, c * 0.27);

  // timing
  const tScale = (i: number, j: number) => PH.scale[0] + (PH.scale[1] - PH.scale[0] - 0.05) * ((i + j) / Math.max(1, 2 * T - 2));
  const tMask = (i: number, j: number) => PH.mask[0] + (PH.mask[1] - PH.mask[0] - 0.08) * ((j - i - 1) / Math.max(1, T - 2));
  const tRow = (i: number) => PH.soft[0] + (PH.soft[1] - PH.soft[0] - 0.06) * (T > 1 ? i / (T - 1) : 0);
  const ROW = { exp: 0, div: 0.018, norm: 0.036, one: 0.05 };

  // per-row softmax pieces (exp of scaled scores and their totals), computed from the real scores
  const rows = useMemo(() => {
    const out: { ex: number[]; sum: number }[] = [];
    for (let i = 0; i < T; i++) {
      const ex: number[] = [];
      let sum = 0;
      for (let j = 0; j < T; j++) {
        const e = j <= i ? Math.exp(S[i * T + j] / 8) : 0;
        ex.push(e);
        sum += e;
      }
      out.push({ ex, sum });
    }
    return out;
  }, [S, T]);

  const scaleRef = useMemo(() => {
    let s = 0;
    for (let n = 0; n < T * T; n++) s += S[n] * S[n];
    return Math.sqrt(s / (T * T)) || 1;
  }, [S, T]);

  const items = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const out: GlyphItem[] = [];
    const white = new THREE.Color(C.token);
    const dim = new THREE.Color(C.dim);
    const kOf = (v: number, sc: number) => 0.55 + 0.5 * Math.min(1, Math.abs(v) / (2 * sc));
    for (let i = 0; i < T; i++) {
      const a = tRow(i);
      for (let j = 0; j < T; j++) {
        const s = S[i * T + j];
        const st: RollState[] = [
          { text: fmtCell(s), t: 0, k: kOf(s, scaleRef) },
          { text: fmtCell(s / 8), t: tScale(i, j), k: kOf(s, scaleRef) },
        ];
        if (j > i) {
          st.push({ text: '−∞', t: tMask(i, j), k: 0.5, color: dim });
          st.push({ text: '0', t: a + ROW.exp, k: 0.32, color: dim });
        } else {
          st.push({ text: fmtCell(rows[i].ex[j]), t: a + ROW.exp, k: 0.75 });
          const p = A[i * T + j];
          st.push({ text: fmt(p), t: a + ROW.norm, k: 0.62 + 0.55 * Math.sqrt(p) });
        }
        out.push(...rollGlyphs(st, cx(j), cy(i), 0.02, { h: gh, color: white, align: 'center' }));
      }
    }
    return out;
  }, [atlas, T, S, A, rows, scaleRef]); // eslint-disable-line react-hooks/exhaustive-deps

  const centers = useMemo(() => {
    const out: [number, number, number][] = [];
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) out.push([cx(j), cy(i), -0.02]);
    return out;
  }, [T, c]); // eslint-disable-line react-hooks/exhaustive-deps

  // hairline frame around the grid
  const frame = useMemo(() => {
    const z = 0;
    const x1 = x0 + W;
    const y1 = y0 - W;
    return [x0, y0, z, x1, y0, z, x1, y0, z, x1, y1, z, x1, y1, z, x0, y1, z, x0, y1, z, x0, y0, z];
  }, [x0, y0, W]);

  // glass sheet over the upper triangle (a staircase polygon) and its outline
  const sheet = useMemo(() => {
    if (T < 2) return null;
    const pts: [number, number][] = [[1, 0], [T, 0], [T, T - 1]];
    for (let r = T - 1; r >= 1; r--) {
      pts.push([r, r]);
      pts.push([r, r - 1]);
    }
    const world = pts.map(([gx, gy]) => new THREE.Vector2(x0 + gx * c, y0 - gy * c));
    const shape = new THREE.Shape(world);
    const geo = new THREE.ShapeGeometry(shape);
    const outline: number[] = [];
    world.forEach((p, n) => {
      const q = world[(n + 1) % world.length];
      outline.push(p.x, p.y, 0, q.x, q.y, 0);
    });
    return { geo, outline };
  }, [T, c, x0, y0]);
  const sheetMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: new THREE.Color(C.norm), transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }),
    [],
  );
  const sheetGroup = useRef<THREE.Group>(null);
  const sheetLine = useRef<THREE.LineSegments>(null);

  const rowBox = useRef<THREE.Group>(null);
  const rowBoxLine = useRef<THREE.LineSegments>(null);
  const rowBoxPts = useMemo(() => {
    const hw = W / 2 + 0.04;
    const hh = c / 2;
    return [-hw, hh, 0, hw, hh, 0, hw, hh, 0, hw, -hh, 0, hw, -hh, 0, -hw, -hh, 0, -hw, -hh, 0, -hw, hh, 0];
  }, [W, c]);
  const cells = useRef<HeatCellsHandle>(null);
  const glyphs = useRef<GlyphFieldHandle>(null);
  const frameRef = useRef<THREE.LineSegments>(null);
  const rowTok = useRef<(TextPlaneHandle | null)[]>([]);
  const colTok = useRef<(TextPlaneHandle | null)[]>([]);
  const titles = useRef<(HTMLDivElement | null)[]>([]);
  const misc = useRef<(HTMLDivElement | null)[]>([]);
  const divL = useRef<(HTMLDivElement | null)[]>([]);
  const oneL = useRef<(HTMLDivElement | null)[]>([]);

  // key labels above the columns: rotate when they would collide
  const colLabelW = useMemo(() => Math.max(...trace.tokens.map((t) => measureText(visibleToken(t.text), tokH))), [trace, tokH]);
  const rotateCols = colLabelW > c * 0.92;

  useStep(index, (p, v) => {
    // sharpened crossfade: this step's objects arrive late and leave early, so neighbours barely overlap
    const e = seg(v, 0.6, 1);
    const vis = e * e * (3 - 2 * e);
    const on = vis > 0.001;
    const gridIn = vis * easeInOut(seg(p, 0, 0.06));
    glyphs.current?.setProgress(p);
    glyphs.current?.setOpacity(on ? gridIn : 0);
    if (frameRef.current) {
      (frameRef.current.material as THREE.LineBasicMaterial).opacity = 0.35 * gridIn;
      frameRef.current.visible = on;
    }
    rowTok.current.forEach((t, i) => t?.setOpacity(gridIn * (i === T - 1 ? 1 : 0.8)));
    colTok.current.forEach((t) => t?.setOpacity(gridIn * 0.8));

    // cell colors
    const cl = cells.current;
    if (cl) {
      for (let i = 0; i < T; i++) {
        const a = tRow(i);
        for (let j = 0; j < T; j++) {
          const n = i * T + j;
          const s = S[n];
          const b = 0.004 + 0.012 * Math.min(1, Math.max(0, 0.5 + s / (4 * scaleRef)));
          let r = WARM[0] * b;
          let g = WARM[1] * b;
          let bl = WARM[2] * b;
          if (j > i) {
            const m = easeInOut(seg(p, tMask(i, j), tMask(i, j) + 0.05));
            const k = 1 - 0.85 * m;
            r *= k;
            g *= k;
            bl *= k;
          } else {
            const hm = easeInOut(seg(p, a + ROW.norm, a + ROW.norm + 0.03));
            const [hr, hg, hb] = heat(A[n]);
            r += (hr - r) * hm;
            g += (hg - g) * hm;
            bl += (hb - bl) * hm;
          }
          cl.set(n, r * gridIn, g * gridIn, bl * gridIn);
        }
      }
      cl.commit(on);
    }

    // glass sheet slides in from the upper right during the mask phase
    const sl = easeInOut(seg(p, PH.mask[0] - 0.02, PH.mask[1] - 0.04));
    if (sheetGroup.current) {
      sheetGroup.current.position.set((1 - sl) * 1.6, (1 - sl) * 1.2, 0.03 + (1 - sl) * 1.2);
      sheetGroup.current.visible = on && sl > 0.001;
    }
    sheetMat.opacity = 0.022 * sl * vis;
    if (sheetLine.current) (sheetLine.current.material as THREE.LineBasicMaterial).opacity = 0.55 * sl * vis;

    // a hairline box follows the row currently being exponentiated and normalized
    if (rowBox.current && rowBoxLine.current) {
      const f = (p - 0.012 - tRow(0)) / Math.max(1e-6, tRow(T - 1) - tRow(0) || 1);
      const u = Math.min(T - 1, Math.max(0, f * (T - 1)));
      // hold on a row while it is processed, glide to the next one in the gap
      const k0 = Math.floor(u);
      const ri = k0 + easeInOut(seg(u - k0, 0.7, 1));
      rowBox.current.position.set(x0 + W / 2, y0 - c * (ri + 0.5), 0.04);
      const a = env(p, tRow(0) - 0.03, tRow(T - 1) + ROW.one + 0.06, 0.03);
      (rowBoxLine.current.material as THREE.LineBasicMaterial).opacity = 0.9 * a * vis;
      rowBoxLine.current.visible = on && a > 0.001;
    }

    // title equation changes with the phase
    const tw = [0, PH.scale[0], PH.mask[0], PH.soft[0], 3];
    titles.current.forEach((el, n) => fade(el, vis * seg(p, tw[n] - 0.02, tw[n] + 0.02) * (1 - seg(p, tw[n + 1] - 0.02, tw[n + 1] + 0.02)) * (n === 0 ? seg(p, 0, 0.04) : 1)));
    fade(misc.current[0], gridIn * 0.9); // shape
    fade(misc.current[1], gridIn * 0.75); // query axis
    fade(misc.current[2], gridIn * 0.75); // key axis
    fade(misc.current[3], vis * seg(p, PH.mask[0] + 0.08, PH.mask[0] + 0.14) * (1 - seg(p, PH.soft[0] - 0.02, PH.soft[0] + 0.03))); // j > i note
    // row totals: ÷ Σ appears after the exponentials, then "= 1" replaces it
    for (let i = 0; i < T; i++) {
      const a = tRow(i);
      fade(divL.current[i], vis * seg(p, a + ROW.div, a + ROW.div + 0.025) * (1 - seg(p, a + ROW.one - 0.015, a + ROW.one)));
      fade(oneL.current[i], vis * seg(p, a + ROW.one, a + ROW.one + 0.025));
    }
  });

  const rightX = x0 + W + 0.35;
  const topY = y0 + (rotateCols ? colLabelW * 0.7 + 0.5 : 0.75);
  return (
    <group>
      <HeatCells ref={cells} centers={centers} size={c} />
      <GlyphField ref={glyphs} items={items} />
      <Lines ref={frameRef} points={frame} color={C.token} opacity={0} />
      <group ref={rowBox}>
        <Lines ref={rowBoxLine} points={rowBoxPts} color={C.attn} opacity={0} />
      </group>
      {sheet && (
        <group ref={sheetGroup}>
          <mesh geometry={sheet.geo} material={sheetMat} renderOrder={6} />
          <Lines ref={sheetLine} points={sheet.outline} color={C.norm} opacity={0} />
        </group>
      )}
      {trace.tokens.map((t, i) => (
        <TextPlane key={`r${i}`} ref={(e) => void (rowTok.current[i] = e)} text={visibleToken(t.text)} height={tokH} position={[x0 - 0.25, cy(i), 0]} anchor="right" />
      ))}
      {trace.tokens.map((t, j) => (
        <group key={`c${j}`} position={[cx(j), y0 + 0.32, 0]} rotation={[0, 0, rotateCols ? Math.PI / 4 : 0]}>
          <TextPlane ref={(e) => void (colTok.current[j] = e)} text={visibleToken(t.text)} height={tokH} position={[0, rotateCols ? 0 : 0.05, 0]} anchor={rotateCols ? 'left' : 'center'} />
        </group>
      ))}
      {[
        'S_h = Q_h K_h^{\\top}',
        'S_h \\,/\\, \\sqrt{64}',
        'S_h / \\sqrt{64} + M',
        'A_h = \\operatorname{softmax}\\!\\big(S_h/\\sqrt{64} + M\\big)',
      ].map((tex, n) => (
        <Label key={n} ref={(e) => void (titles.current[n] = e)} position={[0, topY + 0.55, 0]} tex={tex} size={24} />
      ))}
      <Label ref={(e) => void (misc.current[0] = e)} position={[0, y0 - W - 0.55, 0]} tex={`\\text{head } ${h + 1}\\text{ of block 1}\\qquad ${T}\\times${T}`} size={14} color="#9a9ea6" />
      <Label ref={(e) => void (misc.current[1] = e)} position={[x0 - 0.25, y0 + 0.32, 0]} tex={`\\text{query } i\\ \\downarrow`} size={13} color="#9a9ea6" align="right" />
      <Label ref={(e) => void (misc.current[2] = e)} position={[x0 - 0.25, y0 + 0.75, 0]} tex={`\\text{key } j\\ \\rightarrow`} size={13} color="#9a9ea6" align="right" />
      <Label ref={(e) => void (misc.current[3] = e)} position={[rightX, cy(0), 0]} tex={`j > i:\\ -\\infty`} size={15} color={C.norm} align="left" />
      {rows.map((r, i) => (
        <Label key={`d${i}`} ref={(e) => void (divL.current[i] = e)} position={[rightX, cy(i), 0]} tex={`\\div\\, ${fmtCell(r.sum).replace('−', '-')}`} size={14} color="#c9c2d9" align="left" />
      ))}
      {rows.map((_, i) => (
        <Label key={`o${i}`} ref={(e) => void (oneL.current[i] = e)} position={[rightX, cy(i), 0]} tex={`\\textstyle\\sum = 1`} size={14} color={C.attn} align="left" />
      ))}
    </group>
  );
}
