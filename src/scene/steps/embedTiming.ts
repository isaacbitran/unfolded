// Pure timing of the embedding lookups (shared with tests).

/** Timing for token k of T inside the step: start a, row flight window, etc. (all in step progress p). */
export function tokenWindow(k: number, T: number) {
  const span = 0.6; // p range over which starts are spread
  const a = 0.16 + (T > 1 ? (k * span) / (T - 1) : 0) * 0.86;
  const len = 0.2;
  return {
    idFly: [a, a + len * 0.35] as [number, number],
    band: [a + len * 0.3, a + len * 0.75] as [number, number],
    rowFly: [a + len * 0.55, a + len] as [number, number],
    end: a + len,
  };
}
