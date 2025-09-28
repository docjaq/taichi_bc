from pathlib import Path

path = Path("python/taichi/examples/ggui_examples/BC/mpm3dExt_ggui.py")
lines = path.read_text().splitlines()
start = None
for i, line in enumerate(lines):
    if line.strip() == "if show_pivot_debug:" and lines[i-1].strip().startswith("if show_world_grid"):
        # ensure we're inside show_options block (indent 8 spaces?)
        if not line.startswith("        "):
            continue
        start = i
        break
if start is None:
    raise SystemExit("unwanted pivot block not found")
end = start
while end < len(lines) and lines[end].strip() != "if show_simulation_grid".strip():
    end += 1
# remove block from start to end (exclusive)
lines = lines[:start] + lines[end:]
path.write_text("\n".join(lines) + "\n")
