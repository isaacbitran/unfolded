import { useEffect } from 'react';
import { Stage } from './scene/Stage';
import { StoryPanel } from './ui/StoryPanel';
import { STEP_VH, useScrollStory } from './ui/useScrollStory';
import { useApp } from './state/store';
import { decodeSceneTrace } from './model/trace';
import { STEPS } from './story/steps';
import { PromptBar } from './ui/PromptBar';
import { StoryNav } from './ui/StoryNav';
import { VideoOverlay } from './video/VideoOverlay';

const FILM = new URLSearchParams(location.search).has('video');
import { ExplorePanel } from './explore/ExplorePanel';

export default function App() {
  const mode = useApp((s) => s.mode);
  const trace = useApp((s) => s.trace);
  const reduced = useApp((s) => s.reducedMotion);
  useScrollStory(mode === 'walk' && !FILM);
  useEffect(() => {
    if (new URLSearchParams(location.search).get('mode') === 'explore') useApp.getState().setMode('explore');
    fetch('/demo/demo.trace.bin')
      .then((r) => r.arrayBuffer())
      .then((b) => useApp.getState().setTrace(decodeSceneTrace(b)));
  }, []);
  return (
    <>
      <Stage />
      {FILM && trace && <VideoOverlay />}
      {!trace && <div className="boot">loading GPT-2's numbers…</div>}
      <header className="topbar">
        <div className="brand">
          <span className="mark">Unfolded</span>
          <span className="sub">GPT-2, one operation at a time</span>
        </div>
        <PromptBar />
        <nav className="modes">
          <button className={mode === 'walk' ? 'on' : ''} onClick={() => useApp.getState().setMode('walk')}>Walkthrough</button>
          <button className={mode === 'explore' ? 'on' : ''} onClick={() => useApp.getState().setMode('explore')}>Explore</button>
        </nav>
      </header>
      {mode === 'walk' && <div className="panel-shade" />}
      {mode === 'walk' && <StoryPanel />}
      {mode === 'walk' && <StoryNav />}
      {mode === 'explore' && trace && <ExplorePanel />}
      <footer className="footer">
        <button onClick={() => useApp.getState().setReducedMotion(!useApp.getState().reducedMotion)}>{reduced ? 'Motion: reduced' : 'Motion: full'}</button>
        <a href="https://github.com/isaacbitran/unfolded" target="_blank" rel="noreferrer">Source</a>
      </footer>
      {mode === 'walk' && <div className="scroll-space" style={{ height: `${(STEPS.length + 0.2) * STEP_VH * 100}vh` }} />}
    </>
  );
}
