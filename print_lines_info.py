from pathlib import Path
import re

path = Path("python/taichi/examples/ggui_examples/BC/mpm3dExt_ggui.py")
lines = path.read_text().splitlines()
patterns = OrderedDict([
    ("PIVOT_CONSTANTS", re.compile(r"PIVOT_BBOX_COLOR")),
])
