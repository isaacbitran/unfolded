import { useApp } from '../state/store';
import { S00Intro } from './steps/S00Intro';
import { S01Tokens } from './steps/S01Tokens';
import { S02Embedding } from './steps/S02Embedding';
import { S03Position } from './steps/S03Position';
import { S04Residual } from './steps/S04Residual';
import { S05LayerNorm } from './steps/S05LayerNorm';
import { S06QKV } from './steps/S06QKV';
import { S07Scores } from './steps/S07Scores';
import { S08Softmax } from './steps/S08Softmax';
import { S09Values } from './steps/S09Values';
import { S10Heads } from './steps/S10Heads';
import { S11MLP } from './steps/S11MLP';
import { S12Stack } from './steps/S12Stack';
import { S13Unembed } from './steps/S13Unembed';
import { S14Sampling } from './steps/S14Sampling';
import { S15Autoregress } from './steps/S15Autoregress';
import { S16Training } from './steps/S16Training';

/** Every walkthrough step, mounted once; each hides itself when the story is elsewhere. */
export function World() {
  const trace = useApp((s) => s.trace);
  if (!trace) return null;
  return (
    <>
      <S00Intro index={0} trace={trace} />
      <S01Tokens index={1} trace={trace} />
      <S02Embedding index={2} trace={trace} />
      <S03Position index={3} trace={trace} />
      <S04Residual index={4} trace={trace} />
      <S05LayerNorm index={5} trace={trace} />
      <S06QKV index={6} trace={trace} />
      <S07Scores index={7} trace={trace} />
      <S08Softmax index={8} trace={trace} />
      <S09Values index={9} trace={trace} />
      <S10Heads index={10} trace={trace} />
      <S11MLP index={11} trace={trace} />
      <S12Stack index={12} trace={trace} />
      <S13Unembed index={13} trace={trace} />
      <S14Sampling index={14} trace={trace} />
      <S15Autoregress index={15} trace={trace} />
      <S16Training index={16} trace={trace} />
    </>
  );
}
