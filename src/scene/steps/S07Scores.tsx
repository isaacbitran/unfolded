// Step 7 — attention scores. One head h of block 1 (chosen by `focusHead`: the head whose last
// query row, excluding position 0, has the highest entropy; steps 7–9 share it). The last token's
// 64-entry query q (blue) meets each key k_j (red) in turn: paired entries multiply, the products
// collapse into their sum s_j = q·k_j, and the sum drops into the last row of the T×T score grid.
// An inset draws q and k_j as arrows at their real angle, with a·b = |a||b|cos θ in real numbers.
// Then the grid moves to the center and its other rows fill in, ending where step 8 begins.

import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { SceneTrace } from '../../model/trace';
import { GlyphField, layoutText, useAtlas, type GlyphFieldHandle, type GlyphItem } from '../glyphs/GlyphField';
import { Columns, type ColumnsHandle } from '../primitives/Columns';
import { Lines, bracketPoints } from '../primitives/Glass';
import { Label, fade } from '../primitives/Label';
import { TextPlane, measureText, visibleToken, type TextPlaneHandle } from '../primitives/TextPlane';
import { C, L, easeInOut, fmt, lerp, seg } from '../theme';
import { useStep } from '../useStep';
import { columnBox, glow, rms } from '../vectors';
import { fmtCell, focusHead } from './S08Softmax';

const D = 768;
const DH = 64;
/** Work area (world units). */
const WY = 2.9; // first entry row of q, the key slot and the products
const QX = -5.2;
const KX = -3.35;
const PX = -1.2;
const SUM = { x: 1.3, y: WY - 3.5 * L.rowH };
/** Grid: step 8's layout (cell c, centered at the origin), shown smaller during the dot products. */
const G = { s: 0.68, x: 4.6, y: -1.5 };
/** Key bank above the grid. */
const BANK = { s: 0.55, y0: 3.75 };
/** Inset: the dot product as two arrows. */
const INS = { ox: -4.95, oy: -3.85, len: 2.05 };

const PH = { arrive: [0.02, 0.09] as const, center: [0.585, 0.68] as const, rows: [0.69, 0.94] as const };

/** Per-key timing: the first key is shown slowly, the others quickly. */
function keyWindows(T: number) {
  const out: { a: number; b: number }[] = [];
  let t = 0.08;
  for (let j = 0; j < T; j++) {
    const len = j === 0 ? 0.2 : Math.min(0.055, 0.28 / Math.max(1, T - 1));
    out.push({ a: t, b: t + len });
    t += len + 0.004;
  }
  return out;
}
const phase = (w: { a: number; b: number }, f0: number, f1: number): [number, number] => [w.a + (w.b - w.a) * f0, w.a + (w.b - w.a) * f1];

export function S07Scores({ index, trace }: { index: number; trace: SceneTrace }) {
  const atlas = useAtlas();
  const T = trace.T;
  const L0 = trace.layers[0];
  const h = useMemo(() => focusHead(trace), [trace]);
  const last = T - 1;
  const off = h * DH;
  const S = useMemo(() => L0.scores.subarray(h * T * T, (h + 1) * T * T), [L0, h, T]);
  const win = useMemo(() => keyWindows(T), [T]);
  const keysEnd = win[T - 1].b;

  // step 8's grid geometry (local to the grid group)
  const c = Math.min(1.25, 9.5 / T);
  const W = c * T;
  const gx0 = -W / 2, gy0 = W / 2;
  const cx = (j: number) => gx0 + c * (j + 0.5);
  const cy = (i: number) => gy0 - c * (i + 0.5);
  const gh = Math.min(0.3, c * 0.24);
  const tokH = Math.min(0.32, c * 0.27);
  const toLocal = (x: number, y: number): [number, number, number] => [(x - G.x) / G.s, (y - G.y) / G.s, 0.02];
  const bankX = (j: number) => G.x + G.s * cx(j);

  // real geometry of q and each k_j
  const geo = useMemo(() => {
    const q = L0.q.subarray(last * D + off, last * D + off + DH);
    const nq = Math.hypot(...q);
    const keys = Array.from({ length: T }, (_, j) => {
      const k = L0.k.subarray(j * D + off, j * D + off + DH);
      const nk = Math.hypot(...k);
      const s = S[last * T + j];
      return { nk, s, cos: s / (nq * nk) };
    });
    const maxN = Math.max(nq, ...keys.map((k) => k.nk));
    return { nq, keys, unit: INS.len / maxN };
  }, [L0, last, off, T, S]);

  const qCol = useRef<ColumnsHandle>(null);
  const bank = useRef<(ColumnsHandle | null)[]>([]);
  const work = useRef<GlyphFieldHandle>(null);
  const symbols = useRef<GlyphFieldHandle>(null);
  const gridField = useRef<GlyphFieldHandle>(null);
  const gridGroup = useRef<THREE.Group>(null);
  const gridLines = useRef<THREE.LineSegments>(null);
  const rowBox = useRef<THREE.LineSegments>(null);
  const slotBr = useRef<THREE.LineSegments>(null);
  const prodBr = useRef<THREE.LineSegments>(null);
  const qTok = useRef<TextPlaneHandle>(null);
  const rowTok = useRef<(TextPlaneHandle | null)[]>([]);
  const colTok = useRef<(TextPlaneHandle | null)[]>([]);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const keyLabels = useRef<(HTMLDivElement | null)[]>([]);
  const numLabels = useRef<(HTMLDivElement | null)[]>([]);
  const thetaGroup = useRef<THREE.Group>(null);

  const rowsVis = useMemo(() => [0, 1, 2, 3, 4, 5, -1, DH - 1], []);

  // ---------- moving glyphs of the work area ----------
  const workItems = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const items: GlyphItem[] = [];
    const red = new THREE.Color(C.key);
    const white = new THREE.Color(C.token);
    for (let j = 0; j < T; j++) {
      const w = win[j];
      const fly = phase(w, 0, 0.15);
      const prod = phase(w, 0.16, 0.45);
      const coll = phase(w, 0.47, 0.6);
      const ksc = rms(L0.k, j * D + off, DH);
      const prods: number[] = [];
      for (let d = 0; d < DH; d++) prods.push(L0.q[last * D + off + d] * L0.k[j * D + off + d]);
      let psc = 0;
      for (const v of prods) psc += v * v;
      psc = Math.sqrt(psc / DH) || 1;
      rowsVis.forEach((d, r) => {
        const y = WY - r * L.rowH;
        // the key copy: from its column in the bank to the slot beside q
        const kv = d < 0 ? 0 : L0.k[j * D + off + d];
        const kText = d < 0 ? '⋮' : fmt(kv);
        const kx = d < 0 ? KX : KX + 0.36;
        const bx = bankX(j) + (d < 0 ? 0 : 0.36 * BANK.s);
        const by = BANK.y0 - r * L.rowH * BANK.s;
        items.push(
          ...layoutText(kText, kx, y, 0.03, {
            h: L.glyphH,
            color: red,
            k: d < 0 ? 0.8 : glow(kv, ksc),
            align: d < 0 ? 'center' : 'right',
            from: [bx, by, 0],
            delay: [fly[0] + r * 0.002, fly[1]],
            appear: fly[0] - 0.01,
            vanish: w.b + 0.004,
          }),
        );
        // products: each pair multiplies; the result flies out to the products column…
        const pv = d < 0 ? 0 : prods[d];
        const pText = d < 0 ? '⋮' : fmt(pv);
        const px = d < 0 ? PX : PX + 0.36;
        const t0 = prod[0] + (prod[1] - prod[0]) * 0.6 * (r / 7);
        items.push(
          ...layoutText(pText, px, y, 0.03, {
            h: L.glyphH,
            color: white,
            k: d < 0 ? 0.8 : glow(pv, psc),
            align: d < 0 ? 'center' : 'right',
            from: [(QX + KX) / 2 + 0.2, y, 0],
            delay: [t0, t0 + (prod[1] - prod[0]) * 0.4],
            appear: t0 - 0.006,
            vanish: coll[0] + 0.01,
          }),
        );
        // …and then into the sum
        if (d >= 0) {
          items.push(
            ...layoutText(pText, SUM.x, SUM.y, 0.03, {
              h: L.glyphH,
              color: white,
              k: glow(pv, psc),
              align: 'center',
              from: [px - 0.25, y, 0],
              delay: [coll[0], coll[1]],
              appear: coll[0] - 0.03,
              vanish: coll[1] - 0.004,
            }),
          );
        }
      });
    }
    return items;
  }, [atlas, T, win, L0, off, last, rowsVis]);

  // static symbols in the work area: × and = on every row
  const symbolItems = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const items: GlyphItem[] = [];
    const dim = new THREE.Color(C.dim);
    rowsVis.forEach((d, r) => {
      if (d < 0) return;
      const y = WY - r * L.rowH;
      items.push(...layoutText('×', (QX + KX) / 2 + 0.1, y, 0, { h: 0.24, color: dim, k: 0.9 }));
      items.push(...layoutText('=', (KX + PX) / 2 + 0.1, y, 0, { h: 0.24, color: dim, k: 0.9 }));
    });
    return items;
  }, [atlas, rowsVis]);

  // ---------- grid numbers (grid-local coordinates) ----------
  const rowWin = (i: number) => {
    const n = Math.max(1, T - 1);
    const a = PH.rows[0] + ((PH.rows[1] - PH.rows[0] - 0.04) * i) / n;
    return a;
  };
  const gridItems = useMemo(() => {
    if (!atlas) return [] as GlyphItem[];
    const items: GlyphItem[] = [];
    const white = new THREE.Color(C.token);
    let sc = 0;
    for (let n = 0; n < T * T; n++) sc += S[n] * S[n];
    sc = Math.sqrt(sc / (T * T)) || 1;
    const kOf = (v: number) => 0.55 + 0.5 * Math.min(1, Math.abs(v) / (2 * sc));
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        const s = S[i * T + j];
        if (i === last) {
          // the last row: each score drops in from the work area's sum position
          const w = win[j];
          const coll = phase(w, 0.47, 0.6);
          const drop = phase(w, 0.76, 0.95);
          items.push(
            ...layoutText(fmtCell(s), cx(j), cy(i), 0.02, {
              h: gh,
              color: white,
              k: kOf(s) * 1.1,
              align: 'center',
              from: toLocal(SUM.x, SUM.y),
              delay: drop,
              appear: coll[1] - 0.022,
            }),
          );
        } else {
          const a = rowWin(i) + j * 0.006;
          items.push(
            ...layoutText(fmtCell(s), cx(j), cy(i), 0.02, {
              h: gh,
              color: white,
              k: kOf(s),
              align: 'center',
              from: [cx(j), cy(i) - 0.35, 0.02],
              delay: [a, a + 0.03],
              appear: a - 0.01,
            }),
          );
        }
      }
    }
    return items;
  }, [atlas, T, S, win, last]);

  const gridPts = useMemo(() => {
    const pts: number[] = [];
    for (let n = 0; n <= T; n++) {
      pts.push(gx0, gy0 - c * n, 0, gx0 + W, gy0 - c * n, 0);
      pts.push(gx0 + c * n, gy0, 0, gx0 + c * n, gy0 - W, 0);
    }
    return pts;
  }, [T, c, W, gx0, gy0]);
  const rowBoxPts = useMemo(() => {
    const y1 = cy(last) + c / 2, y0 = cy(last) - c / 2;
    return [gx0, y0, 0.03, gx0 + W, y0, 0.03, gx0, y1, 0.03, gx0 + W, y1, 0.03, gx0, y0, 0.03, gx0, y1, 0.03, gx0 + W, y0, 0.03, gx0 + W, y1, 0.03];
  }, [last, c, gx0, W]);
  const slotPts = useMemo(() => {
    const b = columnBox(KX, WY);
    return bracketPoints(b.x0, b.x1, b.y0, b.y1, 0);
  }, []);
  const prodPts = useMemo(() => {
    const b = columnBox(PX, WY);
    return bracketPoints(b.x0, b.x1, b.y0, b.y1, 0);
  }, []);

  // ---------- inset: q and k_j as arrows at their real angle ----------
  const inset = useMemo(() => {
    const mk = (n: number, color: string, opacity: number) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 6), 3));
      const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
      const l = new THREE.LineSegments(g, m);
      l.frustumCulled = false;
      l.renderOrder = 6;
      l.userData.base = opacity;
      return l;
    };
    const root = new THREE.Group();
    const q = mk(3, C.query, 1);
    const k = mk(3, C.key, 1);
    const proj = mk(24, C.token, 0.45);
    const shadow = mk(1, C.token, 0.95);
    const arc = mk(32, C.token, 0.55);
    const axis = mk(1, C.dim, 0.35);
    [axis, q, k, proj, shadow, arc].forEach((o) => root.add(o));
    return { root, q, k, proj, shadow, arc, axis };
  }, []);

  useStep(index, (p, vis) => {
    const on = vis > 0.001;
    const arr = easeInOut(seg(p, PH.arrive[0], PH.arrive[1]));
    const centerT = easeInOut(seg(p, PH.center[0], PH.center[1]));
    const workA = vis * arr * (1 - seg(p, PH.center[0] - 0.02, PH.center[0] + 0.04));

    qCol.current?.setOpacity(workA);
    qTok.current?.setOpacity(workA);
    symbols.current?.setOpacity(workA * 0.8);
    work.current?.setProgress(p);
    work.current?.setArc(0.5);
    work.current?.setOpacity(on ? vis : 0);
    // bank: the key currently in the slot dims in the bank
    for (let j = 0; j < T; j++) {
      const w = win[j];
      const away = Math.min(seg(p, w.a, w.a + 0.02), 1 - seg(p, w.b - 0.01, w.b + 0.01));
      bank.current[j]?.setOpacity(workA * (1 - 0.7 * away));
      fade(keyLabels.current[j], workA * (1 - 0.5 * away));
    }
    const lineA = (o: THREE.LineSegments | null, a: number) => {
      if (!o) return;
      (o.material as THREE.LineBasicMaterial).opacity = a;
      o.visible = on && a > 0.002;
    };
    lineA(slotBr.current, 0.55 * workA * seg(p, win[0].a, win[0].a + 0.03));
    lineA(prodBr.current, 0.45 * workA * seg(p, phase(win[0], 0.2, 0.3)[0], phase(win[0], 0.2, 0.3)[1]));

    // grid: small beside the work area, then centered at full size where step 8 picks it up
    if (gridGroup.current) {
      const s = lerp(G.s, 1, centerT);
      gridGroup.current.scale.setScalar(s);
      gridGroup.current.position.set(lerp(G.x, 0, centerT), lerp(G.y, 0, centerT), 0);
    }
    gridField.current?.setProgress(p);
    gridField.current?.setArc(0.9);
    gridField.current?.setOpacity(on ? vis : 0);
    lineA(gridLines.current, 0.13 * vis * arr);
    lineA(rowBox.current, 0.75 * vis * seg(p, win[0].a, win[0].a + 0.04) * (1 - seg(p, keysEnd, keysEnd + 0.04)));
    const tokA = vis * seg(p, PH.center[1] - 0.03, PH.center[1] + 0.02);
    rowTok.current.forEach((t, i) => t?.setOpacity(tokA * (i === last ? 1 : 0.8)));
    colTok.current.forEach((t) => t?.setOpacity(tokA * 0.8));

    // labels in the work area
    fade(labels.current[0], workA); // q
    fade(labels.current[1], workA); // 64 under q
    fade(labels.current[2], workA * seg(p, phase(win[0], 0.2, 0.3)[0], phase(win[0], 0.2, 0.3)[1])); // q_d k_jd
    fade(labels.current[3], workA * seg(p, phase(win[0], 0.45, 0.55)[0], phase(win[0], 0.45, 0.55)[1])); // Σ
    fade(labels.current[4], vis * arr * (1 - seg(p, PH.center[0] - 0.02, PH.center[0] + 0.04))); // grid caption (small)
    fade(labels.current[5], vis * seg(p, PH.center[1] - 0.02, PH.center[1] + 0.03)); // title S_h
    fade(labels.current[6], vis * seg(p, PH.center[1] - 0.02, PH.center[1] + 0.03)); // head caption (final)
    fade(labels.current[7], workA); // bank caption
    for (let j = 0; j < T; j++) {
      const w = win[j];
      const a = Math.min(seg(p, w.a, w.a + 0.012), 1 - seg(p, w.b - 0.006, w.b + 0.004));
      fade(numLabels.current[j], workA * (j === T - 1 ? seg(p, w.a, w.a + 0.012) : a));
      fade(numLabels.current[T + j], workA * (j === T - 1 ? seg(p, w.a, w.a + 0.012) : a)); // slot k_j label
    }

    // inset: the red arrow swings to each key's real angle, its length ∝ ‖k_j‖
    const insA = workA * seg(p, 0.04, 0.1);
    fade(labels.current[8], insA);
    fade(labels.current[9], insA);
    fade(labels.current[10], insA);
    fade(labels.current[11], insA);
    let jf = 0;
    for (let j = 1; j < T; j++) jf = Math.max(jf, j - 1 + easeInOut(seg(p, win[j].a - 0.004, win[j].a + 0.02)));
    const j0 = Math.min(T - 1, Math.floor(jf)), j1 = Math.min(T - 1, j0 + 1), f = jf - j0;
    const th = lerp(Math.acos(Math.max(-1, Math.min(1, geo.keys[j0].cos))), Math.acos(Math.max(-1, Math.min(1, geo.keys[j1].cos))), f);
    const lk = lerp(geo.keys[j0].nk, geo.keys[j1].nk, f) * geo.unit;
    const lq = geo.nq * geo.unit;
    const O = [INS.ox, INS.oy];
    const tipQ = [O[0] + lq, O[1]];
    const tipK = [O[0] + lk * Math.cos(th), O[1] + lk * Math.sin(th)];
    const setSegs = (o: THREE.LineSegments, segs: number[][]) => {
      const pos = o.geometry.getAttribute('position') as THREE.BufferAttribute;
      const a = pos.array as Float32Array;
      a.fill(0);
      segs.forEach((s, n) => a.set([s[0], s[1], 0, s[2], s[3], 0], n * 6));
      pos.needsUpdate = true;
    };
    const arrow = (from: number[], to: number[]) => {
      const ang = Math.atan2(to[1] - from[1], to[0] - from[0]);
      const hl = 0.16;
      return [
        [from[0], from[1], to[0], to[1]],
        [to[0], to[1], to[0] - hl * Math.cos(ang - 0.4), to[1] - hl * Math.sin(ang - 0.4)],
        [to[0], to[1], to[0] - hl * Math.cos(ang + 0.4), to[1] - hl * Math.sin(ang + 0.4)],
      ];
    };
    setSegs(inset.q, arrow(O, tipQ));
    setSegs(inset.k, arrow(O, tipK));
    // the shadow of k on q's direction: ‖k‖ cos θ (negative when the angle is obtuse)
    const foot = O[0] + lk * Math.cos(th);
    setSegs(inset.shadow, [[O[0], O[1] - 0.004, foot, O[1] - 0.004]]);
    const dashes: number[][] = [];
    const dl = Math.abs(tipK[1] - O[1]);
    const nd = Math.min(24, Math.max(1, Math.floor(dl / 0.12)));
    for (let n = 0; n < nd; n++) {
      const y0 = O[1] + ((tipK[1] - O[1]) * n) / nd;
      const y1 = O[1] + ((tipK[1] - O[1]) * (n + 0.5)) / nd;
      dashes.push([foot, y0, foot, y1]);
    }
    setSegs(inset.proj, dashes);
    const arcSegs: number[][] = [];
    const ar = 0.42;
    for (let n = 0; n < 32; n++) {
      const a0 = (th * n) / 32, a1 = (th * (n + 1)) / 32;
      arcSegs.push([O[0] + ar * Math.cos(a0), O[1] + ar * Math.sin(a0), O[0] + ar * Math.cos(a1), O[1] + ar * Math.sin(a1)]);
    }
    setSegs(inset.arc, arcSegs);
    setSegs(inset.axis, [[O[0] - 1.25, O[1], O[0], O[1]]]);
    inset.root.children.forEach((o) => {
      const m = (o as THREE.LineSegments).material as THREE.LineBasicMaterial;
      m.opacity = (o.userData.base as number) * insA;
    });
    inset.root.visible = on && insA > 0.002;
    if (thetaGroup.current) thetaGroup.current.position.set(O[0] + 0.68 * Math.cos(th / 2), O[1] + 0.68 * Math.sin(th / 2), 0);
    fade(labels.current[12], insA);
    // arrow tip labels follow the tips
    if (tipGroups.current[0]) tipGroups.current[0].position.set(tipQ[0] + 0.12, tipQ[1] - 0.22, 0);
    if (tipGroups.current[1]) tipGroups.current[1].position.set(tipK[0] + (Math.cos(th) < 0 ? -0.18 : 0.18), tipK[1] + 0.16, 0);
  });
  const tipGroups = useRef<(THREE.Group | null)[]>([]);

  const colLabelW = useMemo(() => Math.max(...trace.tokens.map((t) => measureText(visibleToken(t.text), tokH))), [trace, tokH]);
  const rotateCols = colLabelW > c * 0.92;
  const topY = gy0 + (rotateCols ? colLabelW * 0.7 + 0.5 : 0.75);
  const qxs = useMemo(() => [QX], []);
  const qRows = useMemo(() => [last], [last]);
  const bankXs = useMemo(() => Array.from({ length: T }, (_, j) => [cx(j) * (G.s / BANK.s) + (G.x / BANK.s)]), [T]);
  const bankRows = useMemo(() => Array.from({ length: T }, (_, j) => [j]), [T]);
  const colBottom = columnBox(0, WY).y0;
  const f2 = (v: number) => (v < 0 ? '-' : '') + Math.abs(v).toFixed(2);

  return (
    <group>
      {/* work area */}
      <Columns ref={qCol} data={L0.q} D={D} xs={qxs} rows={qRows} y0={WY} color={C.query} offset={off} len={DH} />
      <TextPlane ref={qTok} text={visibleToken(trace.tokens[last].text)} height={0.32} position={[QX, WY + 0.75, 0]} />
      <GlyphField ref={symbols} items={symbolItems} />
      <Lines ref={slotBr} points={slotPts} color={C.key} opacity={0} />
      <Lines ref={prodBr} points={prodPts} color={C.token} opacity={0} />
      <GlyphField ref={work} items={workItems} />
      <group position={[0, 0, 0]} scale={[BANK.s, BANK.s, 1]}>
        {bankXs.map((xs, j) => (
          <Columns key={j} ref={(e) => void (bank.current[j] = e)} data={L0.k} D={D} xs={xs} rows={bankRows[j]} y0={BANK.y0 / BANK.s} color={C.key} offset={off} len={DH} />
        ))}
      </group>

      {/* grid */}
      <group ref={gridGroup}>
        <Lines ref={gridLines} points={gridPts} color={C.token} opacity={0} />
        <Lines ref={rowBox} points={rowBoxPts} color={C.query} opacity={0} />
        <GlyphField ref={gridField} items={gridItems} />
        {trace.tokens.map((t, i) => (
          <TextPlane key={`r${i}`} ref={(e) => void (rowTok.current[i] = e)} text={visibleToken(t.text)} height={tokH} position={[gx0 - 0.25, cy(i), 0]} anchor="right" />
        ))}
        {trace.tokens.map((t, j) => (
          <group key={`c${j}`} position={[cx(j), gy0 + 0.32, 0]} rotation={[0, 0, rotateCols ? Math.PI / 4 : 0]}>
            <TextPlane ref={(e) => void (colTok.current[j] = e)} text={visibleToken(t.text)} height={tokH} position={[0, rotateCols ? 0 : 0.05, 0]} anchor={rotateCols ? 'left' : 'center'} />
          </group>
        ))}
        <Label ref={(e) => void (labels.current[5] = e)} position={[0, topY + 0.55, 0]} tex="S_h = Q_h K_h^{\top}" size={24} />
        <Label ref={(e) => void (labels.current[6] = e)} position={[0, gy0 - W - 0.55, 0]} tex={`\\text{head } ${h + 1}\\text{ of block 1}\\qquad ${T}\\times${T}`} size={14} color="#9a9ea6" />
        <Label ref={(e) => void (labels.current[4] = e)} position={[0, gy0 - W - 0.7, 0]} tex={`S_h \\in \\mathbb{R}^{${T}\\times ${T}}`} size={15} color="#9a9ea6" />
      </group>
      <primitive object={inset.root} />

      {/* labels */}
      <Label ref={(e) => void (labels.current[0] = e)} position={[QX, colBottom - 0.3, 0]} tex={`q = Q_h[${T}]`} size={16} color={C.query} />
      <Label ref={(e) => void (labels.current[1] = e)} position={[QX, colBottom - 0.66, 0]} tex="64" size={12} color="#8d929a" />
      <Label ref={(e) => void (labels.current[2] = e)} position={[PX, colBottom - 0.3, 0]} tex="q_d\,k_{j,d}" size={16} />
      <Label ref={(e) => void (labels.current[3] = e)} position={[SUM.x - 0.42, SUM.y, 0]} tex="\textstyle\sum_{d=1}^{64} =" size={15} color="#c9c4ba" align="right" />
      <Label ref={(e) => void (labels.current[7] = e)} position={[G.x, BANK.y0 + 0.42, 0]} tex={`K_h \\in \\mathbb{R}^{${T}\\times 64}`} size={15} color={C.key} />
      {trace.tokens.map((_, j) => (
        <Label key={`kb${j}`} ref={(e) => void (keyLabels.current[j] = e)} position={[bankX(j), BANK.y0 - 7 * L.rowH * BANK.s - 0.42, 0]} tex={`k_{${j + 1}}`} size={13} color={C.key} />
      ))}
      {trace.tokens.map((_, j) => (
        <Label key={`ks${j}`} ref={(e) => void (numLabels.current[T + j] = e)} position={[KX, colBottom - 0.3, 0]} tex={`k_{${j + 1}}`} size={16} color={C.key} />
      ))}
      {/* inset labels */}
      <Label ref={(e) => void (labels.current[8] = e)} position={[-1.7, -2.05, 0]} tex="q\cdot k_j = \lVert q\rVert\,\lVert k_j\rVert\cos\theta" size={17} align="left" />
      {geo.keys.map((k, j) => (
        <Label
          key={`n${j}`}
          ref={(e) => void (numLabels.current[j] = e)}
          position={[-1.7, -2.75, 0]}
          tex={`${f2(geo.nq)} \\times ${f2(k.nk)} \\times (${f2(k.cos)}) = ${f2(k.s)}`}
          size={15}
          color="#c9c4ba"
          align="left"
        />
      ))}
      <Label ref={(e) => void (labels.current[9] = e)} position={[-1.7, -3.35, 0]} tex={`\\text{angle sets the sign; lengths set the size}`} size={12} color="#8d929a" align="left" />
      <group ref={(e) => void (tipGroups.current[0] = e)}>
        <Label ref={(e) => void (labels.current[10] = e)} position={[0, 0, 0]} tex="q" size={15} color={C.query} align="left" />
      </group>
      <group ref={(e) => void (tipGroups.current[1] = e)}>
        <Label ref={(e) => void (labels.current[11] = e)} position={[0, 0, 0]} tex="k_j" size={15} color={C.key} />
      </group>
      <group ref={thetaGroup}>
        <Label ref={(e) => void (labels.current[12] = e)} position={[0, 0, 0]} tex="\theta" size={13} color="#c9c4ba" />
      </group>
    </group>
  );
}
