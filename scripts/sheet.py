# Contact sheet of screenshots: python scripts/sheet.py out.png a.png b.png ...
import sys
from PIL import Image
out, files = sys.argv[1], sys.argv[2:]
ims = [Image.open(f).convert('RGB') for f in files]
W, H = ims[0].size
cols = 2
rows = (len(ims) + 1) // 2
sheet = Image.new('RGB', (W * cols, H * rows))
for i, im in enumerate(ims):
    sheet.paste(im.resize((W, H)), ((i % cols) * W, (i // cols) * H))
sheet.resize((W, H * rows // cols)).save(out)
