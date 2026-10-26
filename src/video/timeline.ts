// The demo film: seconds of screen time per walkthrough step, mapped to story position.
// Pure, so the renderer can seek to any frame deterministically.

/** seconds to traverse step i (s from i to i+1); the last step stops at its finished state */
export const STEP_SECONDS = [5.5, 4.2, 8.5, 5.2, 6.0, 6.5, 7.0, 7.0, 6.5, 5.5, 7.0, 7.5, 6.0, 6.5, 7.5, 6.0, 5.5];
export const LAST_S = 16.86;
export const END_HOLD = 4.5;
export const TITLE_SECONDS = 3.2;

export const FILM_SECONDS = STEP_SECONDS.slice(0, -1).reduce((a, b) => a + b, 0) + STEP_SECONDS[16] * (LAST_S - 16) + END_HOLD;

/** Story position at film time t (seconds). */
export function storyAt(t: number): number {
  let acc = 0;
  for (let i = 0; i < STEP_SECONDS.length; i++) {
    const d = STEP_SECONDS[i];
    if (t < acc + d) return Math.min(LAST_S, i + (t - acc) / d);
    acc += d;
  }
  return LAST_S;
}

const smooth = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

/** Opacity of the title card and the end card at time t. */
export function cardsAt(t: number) {
  const title = 1 - smooth((t - TITLE_SECONDS) / 0.9);
  const end = smooth((t - (FILM_SECONDS - END_HOLD + 0.3)) / 1.0);
  return { title, end };
}
