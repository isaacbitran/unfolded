import { useEffect } from 'react';
import { Stage } from './scene/Stage';
import { StoryPanel } from './ui/StoryPanel';
import { STEP_VH, useScrollStory } from './ui/useScrollStory';
import { useApp } from './state/store';
import { decodeSceneTrace } from './model/trace';
import { STEPS } from './story/steps';
import { PromptBar } from './ui/PromptBar';

export default function App() {
  const mode = useApp((s) => s.mode);
  useScrollStory(mode === 'walk');
  useEffect(() => {
    fetch('/demo/demo.trace.bin')
      .then((r) => r.arrayBuffer())
      .then((b) => useApp.getState().setTrace(decodeSceneTrace(b)));
  }, []);
  return (
    <>
      <Stage />
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
      {mode === 'walk' && <div className="scroll-space" style={{ height: `${(STEPS.length + 0.2) * STEP_VH * 100}vh` }} />}
    </>
  );
}
