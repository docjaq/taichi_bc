from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_FILE = SCRIPT_DIR.parent / "mpm3dExt_ggui.py"


def main():
    lines = TARGET_FILE.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "if show_pivot_debug:" and lines[i - 1].strip().startswith("if show_world_grid"):
            if not line.startswith("        "):
                continue
            start = i
            break
    if start is None:
        raise SystemExit("unwanted pivot block not found")
    end = start
    while end < len(lines) and lines[end].strip() != "if show_simulation_grid".strip():
        end += 1
    lines = lines[:start] + lines[end:]
    TARGET_FILE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
