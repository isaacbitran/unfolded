import { useApp } from '../state/store';
import { S02Embedding } from './steps/S02Embedding';

/** Every walkthrough step, mounted once; each hides itself when the story is elsewhere. */
export function World() {
  const trace = useApp((s) => s.trace);
  if (!trace) return null;
  return (
    <>
      <S02Embedding index={2} trace={trace} />
    </>
  );
}
