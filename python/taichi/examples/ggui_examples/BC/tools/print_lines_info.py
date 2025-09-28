from pathlib import Path
from collections import OrderedDict
import re

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_FILE = SCRIPT_DIR.parent / "mpm3dExt_ggui.py"

def main():
    lines = TARGET_FILE.read_text().splitlines()
    patterns = OrderedDict([
        ("PIVOT_CONSTANTS", re.compile(r"PIVOT_BBOX_COLOR")),
    ])

    for label, pattern in patterns.items():
        matches = [i + 1 for i, line in enumerate(lines) if pattern.search(line)]
        print(f"[{label}] matches at lines: {matches}")


if __name__ == "__main__":
    main()
