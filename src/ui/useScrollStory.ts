import { useEffect } from 'react';
import { story } from '../state/store';
import { STEPS } from '../story/steps';

export const STEP_VH = 1.7; // viewport heights of scrolling per step

/** Maps page scroll to story.target; arrow keys / PageUp/PageDown move a whole step. */
export function useScrollStory(active: boolean) {
  useEffect(() => {
    if (!active) return;
    const params = new URLSearchParams(location.search);
    const fixed = params.get('s');
    if (fixed !== null) {
      story.target = story.s = parseFloat(fixed);
      story.frozen = true;
      return;
    }
    const stepPx = () => window.innerHeight * STEP_VH;
    const onScroll = () => (story.target = Math.max(0, Math.min(STEPS.length - 0.001, window.scrollY / stepPx())));
    let anim = 0;
    const goTo = (s: number) => {
      cancelAnimationFrame(anim);
      const from = window.scrollY;
      const to = s * stepPx();
      const t0 = performance.now();
      const dur = Math.min(2600, 900 + Math.abs(to - from) / stepPx() * 1300);
      const tick = (now: number) => {
        const t = Math.min(1, (now - t0) / dur);
        const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        window.scrollTo(0, from + (to - from) * e);
        if (t < 1) anim = requestAnimationFrame(tick);
      };
      anim = requestAnimationFrame(tick);
    };
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input, textarea')) return;
      const cur = Math.floor(story.target + 0.02);
      if (['ArrowDown', 'ArrowRight', 'PageDown', ' '].includes(e.key)) {
        e.preventDefault();
        goTo(Math.min(STEPS.length - 1, cur + 1) + 0.96);
      } else if (['ArrowUp', 'ArrowLeft', 'PageUp'].includes(e.key)) {
        e.preventDefault();
        goTo(Math.max(0, cur - 1) + 0.96);
      }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('keydown', onKey);
    onScroll();
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('keydown', onKey);
    };
  }, [active]);
}
