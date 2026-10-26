// Step 6 — queries, keys and values. GPT-2 stores W_Q, W_K, W_V side by side as one 768 × 2304
// matrix, so they stand here as three adjacent glass slabs. For each token, its normalized column
// flies into the slot left of the slabs, aligned with their 768 rows (one row per input entry); a
// highlight sweeps across the 2304 columns, and as it passes column j the output entry j (the dot
// product of the input with that column) drops into the Q, K or V matrix below. Afterwards each
// 768-entry output column is drawn as a strip of all 768 values and splits into 12 heads of 64.

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { Preview, SceneTrace } from '../../model/trace';
import { GlyphField, layoutText, useAtlas, type GlyphFieldHandle, type GlyphItem } from '../glyphs/GlyphField';
import { Columns, type ColumnsHandle } from '../primitives/Columns';
import { Glass, Lines, bracketPoints, type GlassHandle } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TextPlane, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { C, L, colX, easeInOut, fmt, lerp, seg } from '../theme';
import { useStep } from '../useStep';
import { glow, rms } from '../vectors';
import { COL_TOP } from './S02Embedding';
import { focusHead } from './S08Softmax';

const D = 768;
const NAMES = ['W_Q', 'W_K', 'W_V'] as const;
const OUT = ['Q', 'K', 'V'] as const;
const ROLE = [C.query, C.key, C.value];

/** Slabs: three faces side by side (world units). */
const SL = { w: 4.3, h: 3.9, gap: 0.3, y: 2.9, x0: -6.4, d: 0.24 };
const slabCX = (m: number) => SL.x0 + SL.w / 2 + m * (SL.w + SL.gap);
const SLOT = { sx: (SL.w - 0.5) / 8, sy: (SL.h - 0.4) / 8 };
/** x of face column slot j (0–5 entries, 6 = ⋯, 7 = last) and y of face row slot i. */
const faceX = (m: number, j: number) => slabCX(m) - SL.w / 2 + 0.25 + SLOT.sx * (j + 0.5);
const faceY = (i: number) => SL.y + SL.h / 2 - 0.2 - SLOT.sy * (i + 0.5);
const SLAB_BOTTOM = SL.y - SL.h / 2;
const RIGHT_EDGE = slabCX(2) + SL.w / 2;
/** The input slot left of the slabs. */
const IN_X = SL.x0 - 0.62;

/** Compact matrices: input row and the three output blocks are drawn at 0.7 scale. */
const CS = 0.7;
const COLW = 0.7; // world spacing of token columns inside an output block
const OUT_Y0 = 0.35; // first entry of the output blocks
const IN_Y0 = -3.0; // first entry of the input row (after it has moved down)
const IN_CX = 0.4;
/** Strips (phase 2): 12 segments of 64 values. */
const STRIP = { h: 4.0, gap: 0.1, w: 0.44 };

/** Timing: token 0 is shown slowly, the rest quickly. Sweeps never overlap. */
function tokenWindows(T: number) {
  const out: { fly: [number, number]; sweep: [number, number]; gone: number }[] = [];
  let t = 0.2;
  for (let i = 0; i < T; i++) {
    const len = i === 0 ? 0.17 : Math.min(0.04, 0.2 / Math.max(1, T - 1));
    const flyLen = i === 0 ? 0.05 : 0.03;
    out.push({ fly: [t - flyLen, t], sweep: [t, t + len], gone: t + len + 0.02 });
    t += len + (i === 0 ? 0.02 : 0.003);
  }
  return out;
}
const PH = { arrive: [0.04, 0.13] as const, split: [0.62, 0.69] as const, stretch: [0.69, 0.77] as const, gaps: [0.78, 0.85] as const, focus: [0.88, 0.95] as const };

export function S06QKV({ index, trace }: { index: number; trace: SceneTrace }) {
  const atlas = useAtlas();
  const T = trace.T;
  const L0 = trace.layers[0];
  const outs = useMemo(() => [L0.q, L0.k, L0.v], [L0]);
  const win = useMemo(() => tokenWindows(T), [T]);
  const head = useMemo(() => focusHead(trace), [trace]);
  const lastSweep = win[T - 1].gone;

  const slabs = useRef<(GlassHandle | null)[]>([]);
  const faceRef = useRef<GlyphFieldHandle>(null);
  const inGroup = useRef<THREE.Group>(null);
  const inCols = useRef<ColumnsHandle>(null);
  const inTok = useRef<(TextPlaneHandle | null)[]>([]);
  const travel = useRef<GlyphFieldHandle>(null);
  const outField = useRef<GlyphFieldHandle>(null);
  const outBr = useRef<(THREE.LineSegments | null)[]>([]);
  const band = useRef<THREE.Mesh>(null);
  const slotBr = useRef<THREE.LineSegments>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const outLabelGroups = useRef<(THREE.Group | null)[]>([]);
  const headLabels = useRef<(HTMLDivElement | null)[]>([]);

  // ---------- slab faces: the real corner entries of W_Q, W_K, W_V ----------
  const faceItems = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const items: GlyphItem[] = [];
    const fz = SL.d / 2 + 0.02;
    NAMES.forEach((name, m) => {
      const pv: Preview = trace.previews[name];
      const c = new THREE.Color(ROLE[m]).lerp(new THREE.Color('#ffffff'), 0.5);
      const nr = pv.rows.length, nc = pv.cols.length;
      for (let i = 0; i < nr; i++) {
        const ry = faceY(i < nr - 1 ? i : 7);
        for (let j = 0; j < nc; j++) {
          const cx = faceX(m, j < nc - 1 ? j : 7);
          items.push(...layoutText(fmt(pv.data[i * nc + j]), cx + SLOT.sx * 0.45, ry, fz, { h: 0.17, color: c, k: 0.62, align: 'right' }));
        }
        items.push(...layoutText('⋯', faceX(m, 6), ry, fz, { h: 0.17, color: c, k: 0.45 }));
      }
      for (let j = 0; j < 8; j++) if (j !== 6) items.push(...layoutText('⋮', faceX(m, j) + SLOT.sx * 0.12, faceY(6), fz, { h: 0.17, color: c, k: 0.45 }));
    });
    return items;
  }, [atlas, trace]);

  // ---------- the input copy of token i: flies from the input row into the slot, aligned with slab rows ----------
  const inputX = useMemo(() => trace.tokens.map((_, i) => IN_CX + colX(i, T) * CS), [trace, T]);
  const travelItems = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const items: GlyphItem[] = [];
    const color = new THREE.Color(C.norm);
    for (let i = 0; i < T; i++) {
      const w = win[i];
      const sc = rms(L0.ln1, i * D, D);
      const rows = [0, 1, 2, 3, 4, 5, -1, D - 1];
      rows.forEach((idx, r) => {
        const text = idx < 0 ? '⋮' : fmt(L0.ln1[i * D + idx]);
        const fromX = inputX[i] + (idx < 0 ? 0 : 0.36 * CS);
        const fromY = IN_Y0 - r * L.rowH * CS;
        const tx = idx < 0 ? IN_X : IN_X + 0.3;
        items.push(
          ...layoutText(text, tx, faceY(r), 0.05, {
            h: 0.2,
            color,
            k: idx < 0 ? 0.8 : glow(idx < 0 ? 0 : L0.ln1[i * D + idx], sc) * 1.1,
            align: idx < 0 ? 'center' : 'right',
            from: [fromX, fromY, 0],
            delay: [w.fly[0] + r * 0.002, w.fly[1]],
            appear: w.fly[0] - 0.035,
            vanish: w.gone,
          }),
        );
      });
    }
    return items;
  }, [atlas, T, win, L0, inputX]);

  // ---------- output blocks: entry r of token i's Q/K/V column drops from the slab as the sweep passes ----------
  const sweepT = (i: number, x: number) => {
    const w = win[i].sweep;
    return lerp(w[0], w[1], (x - SL.x0) / (RIGHT_EDGE - SL.x0));
  };
  const outX = (m: number, i: number) => slabCX(m) + (i - (T - 1) / 2) * COLW;
  const outY = (r: number) => OUT_Y0 - r * L.rowH * CS;
  const outItems = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const items: GlyphItem[] = [];
    outs.forEach((data, m) => {
      const color = new THREE.Color(ROLE[m]);
      for (let i = 0; i < T; i++) {
        const sc = rms(data, i * D, D);
        const rows = [0, 1, 2, 3, 4, 5, -1, D - 1];
        rows.forEach((idx, r) => {
          const slot = r; // face column slot of this output entry (6 is ⋯)
          const t0 = sweepT(i, faceX(m, slot));
          const x = outX(m, i);
          const text = idx < 0 ? '⋮' : fmt(data[i * D + idx]);
          const tx = idx < 0 ? x : x + 0.26;
          items.push(
            ...layoutText(text, tx, outY(r), 0, {
              h: 0.22,
              color,
              k: idx < 0 ? 0.8 : glow(data[i * D + idx], sc),
              align: idx < 0 ? 'center' : 'right',
              from: [faceX(m, slot) + 0.2, SLAB_BOTTOM - 0.05, 0.15],
              delay: [t0, t0 + (i === 0 ? 0.045 : 0.02)],
              appear: t0 - 0.012,
              group: m,
            }),
          );
        });
      }
    });
    return items;
  }, [atlas, outs, T, win]);

  // brackets around each output block (one matrix per block) and around the input slot
  const outBrPts = useMemo(
    () =>
      [0, 1, 2].map((m) => {
        const xL = outX(m, 0) - 0.42, xR = outX(m, T - 1) + 0.42;
        return bracketPoints(xL, xR, outY(7) - 0.17, OUT_Y0 + 0.17, 0, 0.12);
      }),
    [T],
  );
  const slotPts = useMemo(() => bracketPoints(IN_X - 0.42, IN_X + 0.42, faceY(7) - 0.2, faceY(0) + 0.2, 0.05, 0.12), []);

  // ---------- strips: all 768 values of every output column, as textures ----------
  const strips = useMemo(() => {
    const segs: THREE.Mesh[][] = [];
    const mats: THREE.MeshBasicMaterial[] = [];
    const focusMats: THREE.MeshBasicMaterial[] = [];
    outs.forEach((data, m) => {
      // texture: T columns of 3 data texels + 1 transparent gap texel; 768 rows (row 0 at the top)
      const W = T * 4 - 1;
      const px = new Uint8Array(W * D * 4);
      const col = new THREE.Color(ROLE[m]);
      for (let i = 0; i < T; i++) {
        const sc = rms(data, i * D, D);
        for (let d = 0; d < D; d++) {
          const b = 0.07 + 0.68 * Math.min(1, Math.abs(data[i * D + d]) / (2.6 * sc));
          for (let x = 0; x < 3; x++) {
            const o = ((D - 1 - d) * W + i * 4 + x) * 4;
            px[o] = Math.round(255 * col.r * b);
            px[o + 1] = Math.round(255 * col.g * b);
            px[o + 2] = Math.round(255 * col.b * b);
            px[o + 3] = 255;
          }
        }
      }
      const tex = new THREE.DataTexture(px, W, D, THREE.RGBAFormat);
      tex.magFilter = THREE.NearestFilter;
      tex.minFilter = THREE.LinearFilter;
      tex.generateMipmaps = false;
      tex.needsUpdate = true;
      const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false, opacity: 0 });
      // the focused head keeps its own material so the others can dim at the end
      const focusMat = mat.clone();
      mats.push(mat);
      focusMats.push(focusMat);
      const row: THREE.Mesh[] = [];
      for (let h = 0; h < 12; h++) {
        const g = new THREE.PlaneGeometry(1, 1);
        // v runs bottom→top in texture space; head h covers dims 64h..64h+63 (rows from the top)
        const v1 = 1 - (h * 64) / D, v0 = 1 - ((h + 1) * 64) / D;
        const uv = g.getAttribute('uv') as THREE.BufferAttribute;
        for (let n = 0; n < uv.count; n++) uv.setY(n, uv.getY(n) > 0.5 ? v1 : v0);
        uv.needsUpdate = true;
        const mesh = new THREE.Mesh(g, h === head ? focusMat : mat);
        mesh.renderOrder = 4;
        row.push(mesh);
      }
      segs.push(row);
    });
    // one memo builds meshes and their parent together (StrictMode runs memos twice)
    const root = new THREE.Group();
    segs.forEach((row) => row.forEach((m) => root.add(m)));
    return { segs, mats, focusMats, root };
  }, [outs, T, head]);
  const stripRoot = strips.root;
  const blockW = (T - 1) * COLW + STRIP.w + 0.02;

  useStep(index, (p, vis) => {
    const on = vis > 0.001;
    // ---- arrival: the six normalized columns shrink into the input row ----
    const arr = easeInOut(seg(p, PH.arrive[0], PH.arrive[1]));
    if (inGroup.current) {
      const s = lerp(1, CS, arr);
      inGroup.current.scale.setScalar(s);
      inGroup.current.position.set(lerp(0, IN_CX, arr), lerp(0, IN_Y0 - CS * COL_TOP, arr), 0);
    }
    const inA = vis * (1 - seg(p, PH.split[0] - 0.04, PH.split[0] + 0.02));
    inCols.current?.setOpacity(inA);
    inTok.current.forEach((t) => t?.setOpacity(inA * seg(p, PH.arrive[1] - 0.03, PH.arrive[1])));
    fade(labels.current[3], inA * seg(p, PH.arrive[1] - 0.03, PH.arrive[1]));

    // ---- slabs ----
    const slabIn = seg(p, 0.08, 0.15);
    const slabDim = 1 - 0.7 * seg(p, PH.split[0], PH.split[1]);
    slabs.current.forEach((s) => s?.setOpacity(vis * slabIn * slabDim));
    faceRef.current?.setOpacity(vis * slabIn * slabDim * 0.85);
    for (let m = 0; m < 3; m++) fade(labels.current[4 + m], vis * slabIn * slabDim);
    fade(labels.current[7], vis * slabIn * slabDim);

    // ---- per-token matrix–vector products ----
    travel.current?.setProgress(p);
    travel.current?.setArc(0.6);
    travel.current?.setOpacity(on ? vis : 0);
    const slotA = vis * seg(p, win[0].fly[0], win[0].fly[1]) * (1 - seg(p, lastSweep - 0.01, lastSweep + 0.02));
    if (slotBr.current) {
      (slotBr.current.material as THREE.LineBasicMaterial).opacity = 0.6 * slotA;
      slotBr.current.visible = slotA > 0.002;
    }
    fade(labels.current[8], slotA);
    // the sweep band: a vertical glow over the slab column currently being dotted with the input
    let bx = 0, bA = 0;
    for (let i = 0; i < T; i++) {
      const w = win[i].sweep;
      if (p >= w[0] - 0.01 && p <= w[1] + 0.01) {
        bx = lerp(SL.x0, RIGHT_EDGE, seg(p, w[0], w[1]));
        bA = Math.min(seg(p, w[0] - 0.01, w[0] + 0.005), 1 - seg(p, w[1] - 0.005, w[1] + 0.01));
      }
    }
    if (band.current) {
      band.current.position.x = bx;
      (band.current.material as THREE.MeshBasicMaterial).opacity = 0.075 * bA * vis;
      band.current.visible = on && bA > 0.002;
    }

    // ---- outputs ----
    outField.current?.setProgress(p);
    outField.current?.setArc(0.4);
    const numA = vis * (1 - seg(p, PH.split[0], PH.split[1]));
    outField.current?.setOpacity(on ? numA : 0);
    outBr.current.forEach((b) => {
      if (!b) return;
      const a = vis * seg(p, win[0].sweep[0], win[0].sweep[0] + 0.03) * (1 - seg(p, PH.split[0], PH.split[1] - 0.03));
      (b.material as THREE.LineBasicMaterial).opacity = 0.6 * a;
      b.visible = a > 0.002;
    });

    // ---- strips: numbers give way to all 768 values, which stretch and split into 12 heads ----
    const sA = vis * seg(p, PH.split[0] + 0.02, PH.split[1]);
    const stretch = easeInOut(seg(p, PH.stretch[0], PH.stretch[1]));
    const gap = easeInOut(seg(p, PH.gaps[0], PH.gaps[1]));
    const focus = easeInOut(seg(p, PH.focus[0], PH.focus[1]));
    const H0 = OUT_Y0 - outY(7) + 0.2; // height of a number column
    const Ht = lerp(H0, STRIP.h, stretch);
    const segH = Ht / 12;
    const g = STRIP.gap * gap;
    const top = OUT_Y0 + 0.12;
    strips.segs.forEach((row, m) => {
      row.forEach((mesh, h) => {
        mesh.scale.set(blockW, segH, 1);
        mesh.position.set(slabCX(m), top - h * (segH + g) - segH / 2, 0);
      });
    });
    // the focused head (picked up by the next step) stays bright while the others dim
    strips.mats.forEach((mat) => (mat.opacity = sA * (1 - 0.72 * focus)));
    strips.focusMats.forEach((mat) => (mat.opacity = sA));
    stripRoot.visible = on && sA > 0.002;
    const yHead = (h: number) => top - h * (segH + g) - segH / 2;
    headLabels.current.forEach((el, h) => {
      const gEl = outLabelGroups.current[3 + h];
      if (gEl) gEl.position.y = yHead(h);
      const a = vis * seg(p, PH.gaps[0] + 0.02, PH.gaps[1]) * (h === head ? 1 : 1 - 0.6 * focus);
      fade(el, a);
    });
    if (focusBox.current) {
      const fb = focusBox.current;
      fb.position.y = yHead(head);
      fb.scale.set(1, segH + 0.08, 1);
      const a = vis * focus;
      (fb.material as THREE.LineBasicMaterial).opacity = 0.9 * a;
      fb.visible = on && a > 0.002;
    }
    const hg = outLabelGroups.current[15];
    if (hg) hg.position.y = yHead(head);
    fade(labels.current[10], vis * focus);
    fade(labels.current[11], vis * seg(p, PH.gaps[0] + 0.02, PH.gaps[1]));

    // output names follow the bottom of whatever is drawn
    const bottom = lerp(outY(7) - 0.17, top - Ht - 11 * g, seg(p, PH.split[0], PH.split[1]) > 0 ? 1 : 0);
    const nameY = Math.min(outY(7) - 0.62, bottom - 0.45);
    for (let m = 0; m < 3; m++) {
      const gEl = outLabelGroups.current[m];
      if (gEl) gEl.position.y = nameY;
      const a = vis * seg(p, win[0].sweep[0], win[0].sweep[0] + 0.04);
      fade(labels.current[m], a * (1 - seg(p, PH.split[0], PH.split[0] + 0.03)));
      fade(labels.current[12 + m], vis * seg(p, PH.gaps[0], PH.gaps[1]));
    }
  });

  const focusBox = useRef<THREE.LineSegments>(null);
  const focusBoxPts = useMemo(() => {
    const x0 = slabCX(0) - blockW / 2 - 0.08, x1 = slabCX(2) + blockW / 2 + 0.08;
    return [x0, -0.5, 0, x1, -0.5, 0, x0, 0.5, 0, x1, 0.5, 0, x0, -0.5, 0, x0, 0.5, 0, x1, -0.5, 0, x1, 0.5, 0];
  }, [blockW]);
  const inXs = useMemo(() => trace.tokens.map((_, i) => colX(i, T)), [trace, T]);

  return (
    <group>
      {[0, 1, 2].map((m) => (
        <Glass key={m} ref={(e) => void (slabs.current[m] = e)} size={[SL.w, SL.h, SL.d]} position={[slabCX(m), SL.y, 0]} color={ROLE[m]} fill={0.004} rim={0.2} edge={0.55} />
      ))}
      <GlyphField ref={faceRef} items={faceItems} />
      <mesh ref={band} position={[0, SL.y, SL.d / 2 + 0.03]} renderOrder={3}>
        <planeGeometry args={[SLOT.sx * 0.55, SL.h]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
      </mesh>
      <group ref={inGroup}>
        <Columns ref={inCols} data={L0.ln1} D={D} xs={inXs} y0={COL_TOP} color={C.norm} />
        {trace.tokens.map((t, i) => (
          <TextPlane key={i} ref={(e) => void (inTok.current[i] = e)} text={visibleToken(t.text)} height={0.32} position={[inXs[i], COL_TOP - 8 * L.rowH - 0.35, 0]} />
        ))}
      </group>
      <GlyphField ref={travel} items={travelItems} />
      <Lines ref={slotBr} points={slotPts} color={C.norm} opacity={0} />
      <GlyphField ref={outField} items={outItems} />
      {outBrPts.map((pts, m) => (
        <Lines key={m} ref={(e) => void (outBr.current[m] = e)} points={pts} color={ROLE[m]} opacity={0} />
      ))}
      <primitive object={stripRoot} />
      <lineSegments ref={focusBox} renderOrder={6}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[new Float32Array(focusBoxPts), 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#ffffff" transparent opacity={0} depthWrite={false} />
      </lineSegments>

      {[0, 1, 2].map((m) => (
        <group key={m} ref={(e) => void (outLabelGroups.current[m] = e)}>
          <Label ref={(e) => void (labels.current[m] = e)} position={[slabCX(m), 0, 0]} tex={`${OUT[m]} \\in \\mathbb{R}^{${T}\\times 768}`} size={17} color={ROLE[m]} />
          <Label ref={(e) => void (labels.current[12 + m] = e)} position={[slabCX(m), 0, 0]} tex={`${OUT[m]} \\in \\mathbb{R}^{12\\times ${T}\\times 64}`} size={17} color={ROLE[m]} />
        </group>
      ))}
      <Label ref={(e) => void (labels.current[3] = e)} position={[IN_CX - (T / 2) * L.colW * CS - 0.3, IN_Y0 - 1.1, 0]} tex={`\\mathrm{LN}(X) \\in \\mathbb{R}^{${T}\\times 768}`} size={15} color={C.norm} align="right" />
      {[0, 1, 2].map((m) => (
        <Label key={m} ref={(e) => void (labels.current[4 + m] = e)} position={[slabCX(m), SL.y + SL.h / 2 + 0.62, 0]} tex={NAMES[m]} size={24} color={ROLE[m]} />
      ))}
      <Label ref={(e) => void (labels.current[7] = e)} position={[slabCX(1), SL.y + SL.h / 2 + 0.22, 0]} tex={`768 \\times 2304 \\text{ in total: three } 768\\times768 \\text{ blocks}`} size={12} color="#9a9ea6" />
      <Label ref={(e) => void (labels.current[8] = e)} position={[IN_X, faceY(7) - 0.55, 0]} tex="768" size={12} color="#9a9ea6" />
      {Array.from({ length: 12 }, (_, h) => (
        <group key={h} ref={(e) => void (outLabelGroups.current[3 + h] = e)}>
          <Label ref={(e) => void (headLabels.current[h] = e)} position={[slabCX(0) - blockW / 2 - 0.22, 0, 0]} tex={`h_{${h + 1}}`} size={13} color={h === head ? '#f4eee2' : '#9a9ea6'} align="right" />
        </group>
      ))}
      <Label ref={(e) => void (labels.current[11] = e)} position={[slabCX(2) + blockW / 2 + 0.25, top64Y(), 0]} tex="64" size={12} color="#9a9ea6" align="left" />
      <group ref={(e) => void (outLabelGroups.current[15] = e)}>
        <Label ref={(e) => void (labels.current[10] = e)} position={[slabCX(2) + blockW / 2 + 0.3, 0, 0]} tex={`\\text{head } ${head + 1}`} size={14} align="left" />
      </group>
    </group>
  );
}

/** y of the first head segment's center once the strips are fully split (for its "64" tag). */
function top64Y() {
  const segH = STRIP.h / 12;
  return OUT_Y0 + 0.12 - segH / 2;
}
