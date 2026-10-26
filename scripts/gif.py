"""Assemble captured frames into an optimised GIF: python scripts/gif.py shots/frames docs/hero.gif [fps] [width]"""
import sys
from pathlib import Path
from PIL import Image

src, out = Path(sys.argv[1]), Path(sys.argv[2])
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 20
width = int(sys.argv[4]) if len(sys.argv) > 4 else 800
frames = []
for f in sorted(src.glob("*.png")):
    im = Image.open(f).convert("RGB")
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
out.parent.mkdir(parents=True, exist_ok=True)
frames[0].save(out, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, optimize=True, disposal=1)
print(f"{len(frames)} frames → {out} ({out.stat().st_size / 1e6:.1f} MB)")
