import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from scene_loader import load_scenes
except Exception as exc:
    load_scenes = None
    print(f"[mpm3dExt] Warning: unable to import scene_loader: {exc}")

AXIS_LABEL_COLOR = (0.95, 0.95, 0.95)
MAX_AXIS_TICK_LABELS = 6
SIM_GRID_MAX_LINES = 64
AXIS_LABEL_CHAR_HEIGHT_FACTOR = 0.0225
AXIS_LABEL_CHAR_ASPECT = 0.6
AXIS_LABEL_CHAR_SPACING = 0.25
AXIS_LABEL_SPACE_FACTOR = 0.65
AXIS_LABEL_DEPTH_OFFSET_FACTOR = 0.006
AXIS_LABEL_OFFSET_FACTOR = 0.035
AXIS_LABEL_X_PADDING = 1.35
AXIS_LABEL_Z_PADDING = 1.2


# Stability configuration
CFL_NUMBER_BASE = 0.5
CFL_NUMBER_EXPONENT = 0.3
CFL_NUMBER_MAX = 0.9
MAX_SUBSTEP_DT_BASE = 8.0e-4

# Simulation configuration (SI units)
DIM = 3
REFERENCE_SIMULATION_LENGTH = 1.0  # metres, canonical domain edge length
SIMULATION_LENGTH = 1.0  # metres, edge length of the cubic domain
SIMULATION_SCALE = max(SIMULATION_LENGTH / REFERENCE_SIMULATION_LENGTH, 1e-6)
MATERIAL_STIFFNESS_SCALE = max(SIMULATION_SCALE, 1.0)
PLASTIC_STRETCH_SCALE = 1.0 / MATERIAL_STIFFNESS_SCALE
STABILITY_SCALE = MATERIAL_STIFFNESS_SCALE
CFL_NUMBER = min(CFL_NUMBER_BASE * (STABILITY_SCALE ** CFL_NUMBER_EXPONENT), CFL_NUMBER_MAX)
MAX_SUBSTEP_DT = MAX_SUBSTEP_DT_BASE / math.sqrt(STABILITY_SCALE)
GRID_RESOLUTION_BASE = 96  # cells along each edge at the reference scale
AUTO_SCALE_GRID_RESOLUTION = True
GRID_RESOLUTION_MAX = 128
GRID_RESOLUTION_MIN = 48
GRID_RESOLUTION_EXPONENT = 0.5
SUBSTEPS_PER_FRAME = 20
PACKING_FRACTION = 0.5  # fraction of cell size used for initial particle spacing
BASE_DENSITY = 1000.0  # kg/m^3
BASE_YOUNG_MODULUS = 1.0e5  # Pa
POISSON_RATIO = 0.2
DEFAULT_GRAVITY = [0.0, -9.81, 0.0]  # m/s^2
GRAVITY = list(DEFAULT_GRAVITY)
BOUNDARY_CELLS = 3

PIVOT_BBOX_COLOR = (0.95, 0.85, 0.2)
PIVOT_AXIS_COLORS = {
    "x": (0.9, 0.3, 0.3),
    "y": (0.3, 0.8, 0.3),
    "z": (0.3, 0.4, 0.9),
}
PIVOT_AXIS_LENGTH_MIN = 0.02 * SIMULATION_LENGTH
PIVOT_AXIS_LENGTH_MAX = 0.2 * SIMULATION_LENGTH

BASE_WATER_STIFFNESS_MULTIPLIER = 1.5
BASE_JELLY_HARDENING = 0.35
BASE_JELLY_STIFFNESS_MULTIPLIER = 0.85
BASE_SNOW_STIFFNESS_MULTIPLIER = 2.2
BASE_CONCRETE_HARDENING_MULTIPLIER = 12.0
BASE_CONCRETE_STIFFNESS_MULTIPLIER = 4.0
CONCRETE_RIGIDITY_GAIN = 6.5
CONCRETE_RIGIDITY_SCALE = math.sqrt(CONCRETE_RIGIDITY_GAIN)
CONCRETE_PLASTIC_COMPRESSIVE = 2.0e-4
CONCRETE_PLASTIC_TENSILE = 5.2e-4

WATER_STIFFNESS_MULTIPLIER = BASE_WATER_STIFFNESS_MULTIPLIER * MATERIAL_STIFFNESS_SCALE
JELLY_HARDENING = BASE_JELLY_HARDENING * math.sqrt(MATERIAL_STIFFNESS_SCALE)
JELLY_STIFFNESS_MULTIPLIER = BASE_JELLY_STIFFNESS_MULTIPLIER * MATERIAL_STIFFNESS_SCALE
SNOW_STIFFNESS_MULTIPLIER = BASE_SNOW_STIFFNESS_MULTIPLIER * MATERIAL_STIFFNESS_SCALE
CONCRETE_HARDENING_MULTIPLIER = BASE_CONCRETE_HARDENING_MULTIPLIER * MATERIAL_STIFFNESS_SCALE * CONCRETE_RIGIDITY_SCALE
CONCRETE_STIFFNESS_MULTIPLIER = BASE_CONCRETE_STIFFNESS_MULTIPLIER * MATERIAL_STIFFNESS_SCALE * CONCRETE_RIGIDITY_SCALE
CONCRETE_MIN_PLASTIC_STRETCH = 1.0 - CONCRETE_PLASTIC_COMPRESSIVE * PLASTIC_STRETCH_SCALE
CONCRETE_MAX_PLASTIC_STRETCH = 1.0 + CONCRETE_PLASTIC_TENSILE * PLASTIC_STRETCH_SCALE

WATER_DAMPING_PER_SECOND = 0.999
JELLY_DAMPING_PER_SECOND = 0.92
SNOW_DAMPING_PER_SECOND = 0.995
CONCRETE_DAMPING_PER_SECOND = 0.992

DEFAULT_PARTICLE_RADIUS = 0.02 * SIMULATION_LENGTH
MAX_PARTICLE_RADIUS = 0.1 * SIMULATION_LENGTH


if AUTO_SCALE_GRID_RESOLUTION:
    scaled_resolution = GRID_RESOLUTION_BASE * (SIMULATION_SCALE ** GRID_RESOLUTION_EXPONENT)
    GRID_RESOLUTION = int(max(GRID_RESOLUTION_MIN, min(GRID_RESOLUTION_MAX, round(scaled_resolution))))
else:
    GRID_RESOLUTION = GRID_RESOLUTION_BASE

dim = DIM
n_grid = GRID_RESOLUTION


dx = SIMULATION_LENGTH / n_grid
BOUNDARY_MARGIN = dx * (BOUNDARY_CELLS + 0.5)
RELATIVE_BOUNDARY_MARGIN = BOUNDARY_MARGIN / SIMULATION_LENGTH
SIDE_MARGIN_REL = max(0.02, RELATIVE_BOUNDARY_MARGIN)

n_particles = n_grid**dim // 2 ** (dim - 1)
print(f"Particles: {n_particles}")
print(f"dx = {dx:.4e} m")
print(f"CFL = {CFL_NUMBER:.2f}")

p_vol = (dx * PACKING_FRACTION) ** dim
p_rho = BASE_DENSITY
p_mass = p_vol * p_rho

effective_stiffness_multipliers = [
    WATER_STIFFNESS_MULTIPLIER,
    JELLY_HARDENING * JELLY_STIFFNESS_MULTIPLIER,
    SNOW_STIFFNESS_MULTIPLIER,
    CONCRETE_HARDENING_MULTIPLIER * CONCRETE_STIFFNESS_MULTIPLIER,
]
max_effective_multiplier = max(effective_stiffness_multipliers)
max_effective_young = BASE_YOUNG_MODULUS * max_effective_multiplier
wave_speed = math.sqrt(max_effective_young / BASE_DENSITY)
raw_dt = CFL_NUMBER * dx / wave_speed
substep_splits = max(1, int(math.ceil(raw_dt / MAX_SUBSTEP_DT)))
dt = raw_dt / substep_splits
steps = SUBSTEPS_PER_FRAME * substep_splits
print(f"dt_raw = {raw_dt:.4e} s (max E = {max_effective_young:.4e} Pa)")
if substep_splits > 1:
    print(f"Using dt = {dt:.4e} s with {steps} substeps per frame (split factor {substep_splits})")
else:
    print(f"Using dt = {dt:.4e} s with {steps} substeps per frame")

def damping_per_step(damping_per_second):
    return damping_per_second**dt


WATER_DAMPING = damping_per_step(WATER_DAMPING_PER_SECOND)
JELLY_DAMPING = damping_per_step(JELLY_DAMPING_PER_SECOND)
SNOW_DAMPING = damping_per_step(SNOW_DAMPING_PER_SECOND)
CONCRETE_DAMPING = damping_per_step(CONCRETE_DAMPING_PER_SECOND)

mu_0 = BASE_YOUNG_MODULUS / (2 * (1 + POISSON_RATIO))
lambda_0 = BASE_YOUNG_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO))  # Lame parameters

def domain_scalar(value):
    return value * SIMULATION_LENGTH


def domain_vector(values):
    return ti.Vector(values) * SIMULATION_LENGTH


def euler_to_rotation_matrix(angles):
    if angles is None:
        return np.identity(3, dtype=np.float32)
    ax, ay, az = angles
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


WORLD_GRID_DIVISIONS = 10


def build_world_grid_geometry(divisions):
    if divisions <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    vertices = []

    def add_line(start_frac, end_frac):
        start = np.array(start_frac, dtype=np.float32) * SIMULATION_LENGTH
        end = np.array(end_frac, dtype=np.float32) * SIMULATION_LENGTH
        vertices.append(start)
        vertices.append(end)

    corners = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    for a, b in edges:
        add_line(corners[a], corners[b])

    for i in range(divisions + 1):
        t = i / divisions
        add_line((0.0, 0.0, t), (1.0, 0.0, t))
        add_line((t, 0.0, 0.0), (t, 0.0, 1.0))

    tick_size = 0.015 * SIMULATION_LENGTH
    tick_frac = tick_size / SIMULATION_LENGTH
    for i in range(1, divisions):
        t = i / divisions
        add_line((t, 0.0, 0.0), (t, 0.0, min(tick_frac, 0.05)))
        add_line((0.0, 0.0, t), (min(tick_frac, 0.05), 0.0, t))

    return np.array(vertices, dtype=np.float32)



def build_axis_line(start_frac, end_frac):
    return np.array([start_frac, end_frac], dtype=np.float32) * SIMULATION_LENGTH



def build_simulation_grid_vertices(resolution, max_lines):
    if resolution <= 0 or max_lines <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    step = max(1, int(math.ceil(resolution / max(max_lines, 1))))
    coords = np.arange(0, resolution + 1, step, dtype=np.int32)
    if coords[-1] != resolution:
        coords = np.append(coords, resolution)
    vertices = []
    scale = SIMULATION_LENGTH / resolution

    def add_line(p0, p1):
        vertices.append(p0)
        vertices.append(p1)

    for y in coords:
        for z in coords:
            start = np.array([0.0, float(y), float(z)], dtype=np.float32) * scale
            end = np.array([float(resolution), float(y), float(z)], dtype=np.float32) * scale
            add_line(start, end)

    for x in coords:
        for z in coords:
            start = np.array([float(x), 0.0, float(z)], dtype=np.float32) * scale
            end = np.array([float(x), float(resolution), float(z)], dtype=np.float32) * scale
            add_line(start, end)

    for x in coords:
        for y in coords:
            start = np.array([float(x), float(y), 0.0], dtype=np.float32) * scale
            end = np.array([float(x), float(y), float(resolution)], dtype=np.float32) * scale
            add_line(start, end)

    return np.array(vertices, dtype=np.float32)

world_grid_vertices_np = build_world_grid_geometry(WORLD_GRID_DIVISIONS)
if world_grid_vertices_np.size > 0:
    world_grid_vertices = ti.Vector.field(3, float, world_grid_vertices_np.shape[0])
    world_grid_vertices.from_numpy(world_grid_vertices_np)
else:
    world_grid_vertices = None

axis_x_vertices_np = build_axis_line((0.0, 0.0, 0.0), (1.05, 0.0, 0.0))
axis_y_vertices_np = build_axis_line((0.0, 0.0, 0.0), (0.0, 1.05, 0.0))
axis_z_vertices_np = build_axis_line((0.0, 0.0, 0.0), (0.0, 0.0, 1.05))

axis_x_vertices = ti.Vector.field(3, float, axis_x_vertices_np.shape[0])
axis_x_vertices.from_numpy(axis_x_vertices_np)

axis_y_vertices = ti.Vector.field(3, float, axis_y_vertices_np.shape[0])
axis_y_vertices.from_numpy(axis_y_vertices_np)

axis_z_vertices = ti.Vector.field(3, float, axis_z_vertices_np.shape[0])
axis_z_vertices.from_numpy(axis_z_vertices_np)
simulation_grid_vertices_np = build_simulation_grid_vertices(n_grid, SIM_GRID_MAX_LINES)
if simulation_grid_vertices_np.size > 0:
    simulation_grid_vertices = ti.Vector.field(3, float, simulation_grid_vertices_np.shape[0])
    simulation_grid_vertices.from_numpy(simulation_grid_vertices_np)
    simulation_grid_line_count = simulation_grid_vertices_np.shape[0] // 2
else:
    simulation_grid_vertices = None
    simulation_grid_line_count = 0

world_grid_tick_labels = [SIMULATION_LENGTH * i / max(WORLD_GRID_DIVISIONS, 1) for i in range(WORLD_GRID_DIVISIONS + 1)]

def select_tick_values(values, max_labels):
    if max_labels <= 0 or len(values) <= max_labels:
        return list(values)
    step = max(1, int(math.ceil((len(values) - 1) / max(max_labels - 1, 1))))
    selected = []
    for idx in range(0, len(values), step):
        selected.append(values[idx])
    if selected[-1] != values[-1]:
        selected.append(values[-1])
    return selected

axis_tick_values = select_tick_values(world_grid_tick_labels, MAX_AXIS_TICK_LABELS)
axis_label_offset = AXIS_LABEL_OFFSET_FACTOR * SIMULATION_LENGTH
axis_label_entries = []


def add_axis_label(position, right, up, text):
    axis_label_entries.append(
        (
            np.array(position, dtype=np.float32),
            np.array(right, dtype=np.float32),
            np.array(up, dtype=np.float32),
            text,
        )
    )


x_face_offset = axis_label_offset * AXIS_LABEL_X_PADDING
z_face_offset = axis_label_offset * AXIS_LABEL_Z_PADDING

for idx, value in enumerate(axis_tick_values):
    label_text = f"{value:.2f}"
    if idx > 0:  # skip duplicate 0 on X axis
        add_axis_label((float(value), axis_label_offset, -x_face_offset), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), label_text)
    add_axis_label((-axis_label_offset, float(value), 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), label_text)
    if idx > 0 or len(axis_tick_values) == 1:
        add_axis_label((-z_face_offset, axis_label_offset, float(value)), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), label_text)

add_axis_label((SIMULATION_LENGTH * 1.04, axis_label_offset, -x_face_offset), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "X")
add_axis_label((-axis_label_offset, SIMULATION_LENGTH * 1.04, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "Y")
add_axis_label((-z_face_offset, axis_label_offset, SIMULATION_LENGTH * 1.04), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "Z")

SEGMENT_LINES = {
    "A": ((0.0, 1.0), (1.0, 1.0)),
    "B": ((1.0, 1.0), (1.0, 0.5)),
    "C": ((1.0, 0.5), (1.0, 0.0)),
    "D": ((0.0, 0.0), (1.0, 0.0)),
    "E": ((0.0, 0.5), (0.0, 0.0)),
    "F": ((0.0, 1.0), (0.0, 0.5)),
    "G": ((0.0, 0.5), (1.0, 0.5)),
}

DIGIT_SEGMENT_MAP = {
    "0": ("A", "B", "C", "D", "E", "F"),
    "1": ("B", "C"),
    "2": ("A", "B", "G", "E", "D"),
    "3": ("A", "B", "G", "C", "D"),
    "4": ("F", "G", "B", "C"),
    "5": ("A", "F", "G", "C", "D"),
    "6": ("A", "F", "E", "D", "C", "G"),
    "7": ("A", "B", "C"),
    "8": ("A", "B", "C", "D", "E", "F", "G"),
    "9": ("A", "B", "C", "D", "F", "G"),
}

STROKE_FONT = {digit: [SEGMENT_LINES[s] for s in segments] for digit, segments in DIGIT_SEGMENT_MAP.items()}
STROKE_FONT["."] = [
    ((0.75, 0.05), (0.9, 0.05)),
    ((0.9, 0.05), (0.9, 0.0)),
    ((0.9, 0.0), (0.75, 0.0)),
    ((0.75, 0.0), (0.75, 0.05)),
]
STROKE_FONT["X"] = [((0.0, 0.0), (1.0, 1.0)), ((0.0, 1.0), (1.0, 0.0))]
STROKE_FONT["Y"] = [((0.0, 1.0), (0.5, 0.5)), ((1.0, 1.0), (0.5, 0.5)), ((0.5, 0.5), (0.5, 0.0))]
STROKE_FONT["Z"] = [((0.0, 1.0), (1.0, 1.0)), ((1.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (1.0, 0.0))]

AXIS_LABEL_CHAR_HEIGHT = AXIS_LABEL_CHAR_HEIGHT_FACTOR * SIMULATION_LENGTH
AXIS_LABEL_CHAR_WIDTH = AXIS_LABEL_CHAR_HEIGHT * AXIS_LABEL_CHAR_ASPECT
AXIS_LABEL_GLYPH_SPACING = AXIS_LABEL_CHAR_WIDTH * AXIS_LABEL_CHAR_SPACING
AXIS_LABEL_SPACE_ADVANCE = AXIS_LABEL_CHAR_WIDTH * AXIS_LABEL_SPACE_FACTOR
AXIS_LABEL_DEPTH_OFFSET = AXIS_LABEL_DEPTH_OFFSET_FACTOR * SIMULATION_LENGTH


def build_axis_label_geometry(entries):
    if AXIS_LABEL_CHAR_HEIGHT <= 0 or not entries:
        return np.zeros((0, 3), dtype=np.float32)

    vertices = []
    char_height = AXIS_LABEL_CHAR_HEIGHT
    char_width = AXIS_LABEL_CHAR_WIDTH
    glyph_spacing = AXIS_LABEL_GLYPH_SPACING
    space_advance = AXIS_LABEL_SPACE_ADVANCE
    depth_bias = AXIS_LABEL_DEPTH_OFFSET

    for position, right_vec, up_vec, raw_text in entries:
        text = raw_text.upper()
        right_norm = np.linalg.norm(right_vec)
        up_norm = np.linalg.norm(up_vec)
        if right_norm < 1e-6 or up_norm < 1e-6:
            continue
        right_dir = right_vec / right_norm
        up_dir = up_vec / up_norm
        normal = np.cross(right_dir, up_dir)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            depth_offset = np.zeros(3, dtype=np.float32)
        else:
            depth_offset = normal / normal_norm * depth_bias

        text_width = 0.0
        first = True
        for ch in text:
            if ch == " ":
                text_width += space_advance
                continue
            if ch not in STROKE_FONT:
                text_width += space_advance
                continue
            if not first:
                text_width += glyph_spacing
            text_width += char_width
            first = False

        if text_width <= 1e-6:
            continue

        origin = position.astype(np.float32) + depth_offset - right_dir * (text_width * 0.5) - up_dir * (char_height * 0.5)
        cursor = 0.0
        first = True

        for ch in text:
            if ch == " ":
                cursor += space_advance
                continue
            glyph = STROKE_FONT.get(ch)
            if glyph is None:
                cursor += space_advance
                continue
            if not first:
                cursor += glyph_spacing
            for (sx, sy), (ex, ey) in glyph:
                start = origin + right_dir * (cursor + sx * char_width) + up_dir * (sy * char_height)
                end = origin + right_dir * (cursor + ex * char_width) + up_dir * (ey * char_height)
                vertices.append(start.astype(np.float32))
                vertices.append(end.astype(np.float32))
            cursor += char_width
            first = False

    if not vertices:
        return np.zeros((0, 3), dtype=np.float32)

    return np.stack(vertices, axis=0)


axis_label_vertices_np = build_axis_label_geometry(axis_label_entries)
if axis_label_vertices_np.size > 0:
    axis_label_vertices = ti.Vector.field(3, float, axis_label_vertices_np.shape[0])
    axis_label_vertices.from_numpy(axis_label_vertices_np)
    axis_label_line_count = axis_label_vertices_np.shape[0] // 2
else:
    axis_label_vertices = None
    axis_label_line_count = 0

F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
F_dg = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)  # deformation gradient
F_Jp = ti.field(float, n_particles)

F_colors = ti.Vector.field(4, float, n_particles)
F_colors_random = ti.Vector.field(4, float, n_particles)
F_materials = ti.field(int, n_particles)
F_grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)
F_grid_m = ti.field(float, (n_grid,) * dim)
F_used = ti.field(int, n_particles)

neighbour = (3,) * dim

WATER = 0
JELLY = 1
SNOW = 2
CONCRETE = 3


MATERIAL_NAME_MAP: Dict[str, int] = {
    "WATER": WATER,
    "JELLY": JELLY,
    "SNOW": SNOW,
    "CONCRETE": CONCRETE,
}


def _resolve_material(material_name: str, warning_sink: List[str]) -> int:
    key = material_name.upper()
    if key not in MATERIAL_NAME_MAP:
        warning_sink.append(f"unknown material '{material_name}', defaulting to WATER")
    return MATERIAL_NAME_MAP.get(key, WATER)


def _sanitize_override_vec3(value: object, name: str, warning_sink: List[str]) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        warning_sink.append(f"override '{name}' must be a sequence of 3 numbers; ignored")
        return None
    try:
        vec = tuple(float(v) for v in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        warning_sink.append(f"override '{name}' has non-numeric entries; ignored")
        return None
    return vec  # type: ignore[return-value]


def _scene_object_summary(scene_object) -> str:
    material = getattr(scene_object, "material", "?")
    ident = getattr(scene_object, "scene_id", getattr(scene_object, "id", "object"))
    pivot = getattr(scene_object, "pivot", (0.5, 0.0, 0.5))
    shape_type = getattr(scene_object, "obj_type", getattr(scene_object, "shape_type", "?"))
    extra = ""
    if shape_type == "sphere":
        radius = getattr(scene_object, "radius", None)
        if radius is not None:
            try:
                extra = f" r={float(radius):.3f}"
            except Exception:
                extra = ""
    elif shape_type == "cube_volume":
        size = getattr(scene_object, "size", None)
        if size is not None:
            try:
                sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
                extra = f" size=({sx:.3f}, {sy:.3f}, {sz:.3f})"
            except Exception:
                extra = ""
    return f"{ident} [{material}] {shape_type}{extra} pivot={pivot}"



def vector_to_np(vec) -> np.ndarray:
    return np.array([float(vec[0]), float(vec[1]), float(vec[2])], dtype=np.float32)


def build_pivot_debug_geometry(scene_entry) -> tuple[Optional[np.ndarray], dict[str, Optional[np.ndarray]]]:
    bbox_vertices: list[np.ndarray] = []
    axis_vertices = {"x": [], "y": [], "z": []}
    if scene_entry is None:
        return None, {"x": None, "y": None, "z": None}

    for shape in scene_entry.shapes:
        size_attr = getattr(shape, "size", None)
        minimum_attr = getattr(shape, "minimum", None)
        if size_attr is None or minimum_attr is None:
            continue
        size_vec = vector_to_np(size_attr)
        if not np.all(size_vec > 0):
            continue
        min_vec = vector_to_np(minimum_attr)
        pivot_rel = np.array(getattr(shape, "pivot", (0.5, 0.5, 0.5)), dtype=np.float32)
        rotation = euler_to_rotation_matrix(getattr(shape, "rotation", (0.0, 0.0, 0.0)))
        pivot_world = min_vec + size_vec * pivot_rel

        corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        corners_world = []
        for corner in corners:
            local = (corner - pivot_rel) * size_vec
            world = rotation @ local + pivot_world
            corners_world.append(world.astype(np.float32))

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            bbox_vertices.append(corners_world[a])
            bbox_vertices.append(corners_world[b])

        axis_length = max(size_vec.max() * 0.3, PIVOT_AXIS_LENGTH_MIN)
        axis_length = min(axis_length, PIVOT_AXIS_LENGTH_MAX)
        if axis_length <= 0:
            axis_length = PIVOT_AXIS_LENGTH_MIN

        axis_local = [
            np.array([axis_length, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, axis_length, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, axis_length], dtype=np.float32),
        ]
        for key, direction in zip(("x", "y", "z"), axis_local):
            end = rotation @ direction + pivot_world
            axis_vertices[key].append(pivot_world.astype(np.float32))
            axis_vertices[key].append(end.astype(np.float32))

    if not bbox_vertices:
        return None, {"x": None, "y": None, "z": None}

    bbox_np = np.array(bbox_vertices, dtype=np.float32)
    axis_np = {
        key: (np.array(values, dtype=np.float32) if values else None)
        for key, values in axis_vertices.items()
    }
    return bbox_np, axis_np



def rebuild_pivot_debug_geometry():
    global pivot_bbox_vertices, pivot_axis_vertices
    global pivot_bbox_vertex_count, pivot_axis_vertex_counts
    if not scene_entries or curr_scene_id >= len(scene_entries):
        pivot_bbox_vertices = None
        pivot_axis_vertices = {"x": None, "y": None, "z": None}
        pivot_bbox_vertex_count = 0
        pivot_axis_vertex_counts = {"x": 0, "y": 0, "z": 0}
        return

    bbox_np, axis_np = build_pivot_debug_geometry(scene_entries[curr_scene_id])

    if bbox_np is not None and bbox_np.size > 0:
        pivot_bbox_vertex_count = bbox_np.shape[0]
        pivot_bbox_vertices = ti.Vector.field(3, float, pivot_bbox_vertex_count)
        pivot_bbox_vertices.from_numpy(bbox_np)
    else:
        pivot_bbox_vertices = None
        pivot_bbox_vertex_count = 0

    axis_fields: Dict[str, Optional[Any]] = {}
    axis_counts: Dict[str, int] = {}
    for axis_key in ("x", "y", "z"):
        axis_arr = None
        if axis_np:
            axis_arr = axis_np.get(axis_key)
        if axis_arr is not None and axis_arr.size > 0:
            axis_counts[axis_key] = axis_arr.shape[0]
            axis_field = ti.Vector.field(3, float, axis_arr.shape[0])
            axis_field.from_numpy(axis_arr)
            axis_fields[axis_key] = axis_field
        else:
            axis_counts[axis_key] = 0
            axis_fields[axis_key] = None

    pivot_axis_vertices = axis_fields
    pivot_axis_vertex_counts = axis_counts


def _build_scene_entry_from_definition(definition) -> Optional["SceneEntry"]:
    warnings = list(getattr(definition, "warnings", []))
    shapes: List[SceneShape] = []
    summaries: List[str] = []

    for obj in getattr(definition, "objects", []):
        material_id = _resolve_material(getattr(obj, "material", "WATER"), warnings)
        obj_type = getattr(obj, "obj_type", "cube_volume")
        if obj_type == "sphere":
            center_data = getattr(obj, "center", None)
            radius_data = getattr(obj, "radius", None)
            if center_data is None or radius_data is None:
                warnings.append(f"sphere '{getattr(obj, 'scene_id', '?')}' missing center or radius; skipped")
                continue
            try:
                center = domain_vector(center_data)
                radius = domain_scalar(radius_data)
            except Exception:
                warnings.append(f"sphere '{getattr(obj, 'scene_id', '?')}' failed to convert center/radius; skipped")
                continue
            rotation = getattr(obj, "rotation_euler", (0.0, 0.0, 0.0))
            velocity = getattr(obj, "initial_velocity", (0.0, 0.0, 0.0))
            pivot = getattr(obj, "pivot", (0.5, 0.5, 0.5))
            shapes.append(
                SphereVolume(
                    center=center,
                    radius=radius,
                    material=material_id,
                    rotation=rotation,
                    initial_velocity=velocity,
                    pivot=pivot,
                    color_override=getattr(obj, "color_override", None),
                )
            )
            summaries.append(_scene_object_summary(obj))
            continue

        try:
            minimum = domain_vector(getattr(obj, "position"))
            size = domain_vector(getattr(obj, "size"))
        except Exception:
            warnings.append(f"object '{getattr(obj, 'scene_id', '?')}' has invalid position/size; skipped")
            continue
        rotation = getattr(obj, "rotation_euler", (0.0, 0.0, 0.0))
        velocity = getattr(obj, "initial_velocity", (0.0, 0.0, 0.0))
        shapes.append(
            CubeVolume(
                minimum=minimum,
                size=size,
                material=material_id,
                rotation=rotation,
                initial_velocity=velocity,
                pivot=getattr(obj, "pivot", (0.5, 0.0, 0.5)),
                color_override=getattr(obj, "color_override", None),
            )
        )
        summaries.append(_scene_object_summary(obj))

    overrides: Dict[str, object] = {}
    override_payload = getattr(definition, "simulation_overrides", {})
    if isinstance(override_payload, dict):
        gravity_override = _sanitize_override_vec3(override_payload.get("gravity"), "gravity", warnings)
        if gravity_override is not None:
            overrides["gravity"] = gravity_override
        timeline_duration = override_payload.get("timeline_duration")
        if timeline_duration is not None:
            try:
                overrides["timeline_duration"] = float(timeline_duration)
            except (TypeError, ValueError):
                warnings.append("override 'timeline_duration' must be numeric; ignored")

    source = ""
    path_obj = getattr(definition, "path", None)
    if path_obj is not None:
        try:
            source = str(Path(path_obj).resolve().relative_to(CURRENT_DIR))
        except Exception:
            source = str(path_obj)

    title = getattr(definition, "title", getattr(definition, "key", "Scene"))
    description = getattr(definition, "description", "")

    if not shapes:
        warnings.append("scene contains no usable objects")
        return None

    return SceneEntry(
        title=title,
        description=description,
        shapes=shapes,
        overrides=overrides,
        warnings=warnings,
        source=source or "(inline)",
        object_summaries=summaries,
    )




@ti.kernel
def substep(g_x: float, g_y: float, g_z: float):
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        F_dg[p] = (ti.Matrix.identity(float, 3) + dt * F_C[p]) @ F_dg[p]  # deformation gradient update
        # Hardening coefficient: snow gets harder when compressed
        h = ti.exp(10 * (1.0 - F_Jp[p]))
        stiffness_multiplier = WATER_STIFFNESS_MULTIPLIER
        mat = F_materials[p]
        if mat == JELLY:  # jelly, very soft and damped
            h = JELLY_HARDENING
            stiffness_multiplier = JELLY_STIFFNESS_MULTIPLIER
        elif mat == CONCRETE:  # concrete, stiffer
            h = CONCRETE_HARDENING_MULTIPLIER
            stiffness_multiplier = CONCRETE_STIFFNESS_MULTIPLIER
        elif mat == SNOW:
            stiffness_multiplier = SNOW_STIFFNESS_MULTIPLIER
        mu, la = mu_0 * h * stiffness_multiplier, lambda_0 * h * stiffness_multiplier
        if mat == WATER:  # liquid
            mu = 0.0

        U, sig, V = ti.svd(F_dg[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            if mat == SNOW:  # Snow
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            elif mat == CONCRETE:  # Concrete behaves rigidly with tiny plasticity
                new_sig = ti.min(
                    ti.max(sig[d, d], CONCRETE_MIN_PLASTIC_STRETCH),
                    CONCRETE_MAX_PLASTIC_STRETCH,
                )
            F_Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if mat == WATER:
            # Reset deformation gradient to avoid numerical instability
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = J
            F_dg[p] = new_F
        elif mat == SNOW or mat == CONCRETE:
            # Reconstruct elastic deformation gradient after plasticity
            F_dg[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F_dg[p] - U @ V.transpose()) @ F_dg[p].transpose() + ti.Matrix.identity(
            float, 3
        ) * la * J * (J - 1)
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + p_mass * F_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        cond = (I < BOUNDARY_CELLS) & (F_grid_v[I] < 0) | (I > n_grid - BOUNDARY_CELLS) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        damping = WATER_DAMPING
        mat = F_materials[p]
        if mat == JELLY:
            damping = JELLY_DAMPING
        elif mat == SNOW:
            damping = SNOW_DAMPING
        elif mat == CONCRETE:
            damping = CONCRETE_DAMPING
        F_v[p] = new_v * damping
        F_x[p] += dt * F_v[p]
        F_C[p] = new_C


class CubeVolume:
    def __init__(self, minimum, size, material, rotation=None, initial_velocity=None, pivot=None, color_override=None):
        self.shape_type = "cube_volume"
        self.minimum = minimum
        self.size = size
        self.material = material
        self.rotation = tuple(rotation) if rotation is not None else (0.0, 0.0, 0.0)
        self.initial_velocity = tuple(initial_velocity) if initial_velocity is not None else (0.0, 0.0, 0.0)
        if pivot is None:
            pivot = (0.5, 0.0, 0.5)
        self.pivot = tuple(pivot)
        self.color_override = color_override
        self.volume = float(self.size.x * self.size.y * self.size.z)


class SphereVolume:
    def __init__(self, center, radius, material, rotation=None, initial_velocity=None, pivot=None, color_override=None):
        self.shape_type = "sphere"
        self.center = center
        self.radius = float(radius)
        self.material = material
        self.rotation = tuple(rotation) if rotation is not None else (0.0, 0.0, 0.0)
        self.initial_velocity = tuple(initial_velocity) if initial_velocity is not None else (0.0, 0.0, 0.0)
        if pivot is None:
            pivot = (0.5, 0.5, 0.5)
        self.pivot = tuple(pivot)
        self.color_override = color_override
        diameter = self.radius * 2.0
        self.size = ti.Vector([diameter, diameter, diameter])
        radius_vec = ti.Vector([self.radius, self.radius, self.radius])
        self.minimum = center - radius_vec
        self.volume = float((4.0 / 3.0) * math.pi * (self.radius ** 3))


SceneShape = Union["CubeVolume", "SphereVolume"]



@dataclass
class SceneEntry:
    title: str
    description: str
    shapes: List["SceneShape"]
    overrides: Dict[str, object]
    warnings: List[str]
    source: str
    object_summaries: List[str]



@ti.kernel
def init_cube_vol(
    first_par: int,
    last_par: int,
    x_begin: float,
    y_begin: float,
    z_begin: float,
    x_size: float,
    y_size: float,
    z_size: float,
    material: int,
    r00: float,
    r01: float,
    r02: float,
    r10: float,
    r11: float,
    r12: float,
    r20: float,
    r21: float,
    r22: float,
    vel_x: float,
    vel_y: float,
    vel_z: float,
    pivot_x: float,
    pivot_y: float,
    pivot_z: float,
):
    rotation = ti.Matrix([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    min_vec = ti.Vector([x_begin, y_begin, z_begin])
    size_vec = ti.Vector([x_size, y_size, z_size])
    pivot_vec = ti.Vector([pivot_x, pivot_y, pivot_z])
    pivot_world = min_vec + size_vec * pivot_vec
    for i in range(first_par, last_par):
        random_vec = ti.Vector([ti.random() for _ in range(dim)])
        local = (random_vec - pivot_vec) * size_vec
        rotated = rotation @ local
        F_x[i] = rotated + pivot_world
        F_Jp[i] = 1
        F_dg[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_v[i] = ti.Vector([vel_x, vel_y, vel_z])
        F_materials[i] = material
        F_colors_random[i] = ti.Vector([ti.random(), ti.random(), ti.random(), ti.random()])
        F_used[i] = 1

@ti.kernel
def init_sphere(
    first_par: int,
    last_par: int,
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float,
    material: int,
    vel_x: float,
    vel_y: float,
    vel_z: float,
):
    center = ti.Vector([center_x, center_y, center_z])
    for i in range(first_par, last_par):
        cos_phi = 2.0 * ti.random(dtype=float) - 1.0
        theta = 2.0 * math.pi * ti.random(dtype=float)
        sin_phi = ti.sqrt(ti.max(0.0, 1.0 - cos_phi * cos_phi))
        direction = ti.Vector([
            sin_phi * ti.cos(theta),
            sin_phi * ti.sin(theta),
            cos_phi,
        ])
        radius_scale = ti.pow(ti.random(dtype=float), 1.0 / 3.0)
        position = center + direction * (radius * radius_scale)
        F_x[i] = position
        F_Jp[i] = 1
        F_dg[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_v[i] = ti.Vector([vel_x, vel_y, vel_z])
        F_materials[i] = material
        F_colors_random[i] = ti.Vector([ti.random(), ti.random(), ti.random(), ti.random()])
        F_used[i] = 1




@ti.kernel
def set_all_unused():
    for p in F_used:
        F_used[p] = 0
        # basically throw them away so they aren't rendered
        F_x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        F_Jp[p] = 1
        F_dg[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        F_v[p] = ti.Vector([0.0, 0.0, 0.0])


def init_vols(shapes: List[SceneShape]):
    set_all_unused()
    if not shapes:
        return

    volume_weights = [max(getattr(shape, "volume", 0.0), 0.0) for shape in shapes]
    total_weight = sum(volume_weights)
    if total_weight <= 0.0:
        volume_weights = [1.0 for _ in shapes]
        total_weight = float(len(shapes))

    next_p = 0
    for index, shape in enumerate(shapes):
        if total_weight > 0.0:
            par_count = int(volume_weights[index] / total_weight * n_particles)
        else:
            par_count = n_particles // len(shapes)
        if index == len(shapes) - 1:
            par_count = n_particles - next_p
        if par_count <= 0:
            continue

        if isinstance(shape, CubeVolume):
            rot_mat = euler_to_rotation_matrix(shape.rotation)
            init_cube_vol(
                next_p,
                next_p + par_count,
                float(shape.minimum[0]),
                float(shape.minimum[1]),
                float(shape.minimum[2]),
                float(shape.size[0]),
                float(shape.size[1]),
                float(shape.size[2]),
                shape.material,
                float(rot_mat[0, 0]),
                float(rot_mat[0, 1]),
                float(rot_mat[0, 2]),
                float(rot_mat[1, 0]),
                float(rot_mat[1, 1]),
                float(rot_mat[1, 2]),
                float(rot_mat[2, 0]),
                float(rot_mat[2, 1]),
                float(rot_mat[2, 2]),
                float(shape.initial_velocity[0]),
                float(shape.initial_velocity[1]),
                float(shape.initial_velocity[2]),
                float(shape.pivot[0]),
                float(shape.pivot[1]),
                float(shape.pivot[2]),
            )
        elif isinstance(shape, SphereVolume):
            init_sphere(
                next_p,
                next_p + par_count,
                float(shape.center[0]),
                float(shape.center[1]),
                float(shape.center[2]),
                float(shape.radius),
                shape.material,
                float(shape.initial_velocity[0]),
                float(shape.initial_velocity[1]),
                float(shape.initial_velocity[2]),
            )
        else:
            continue

        next_p += par_count


@ti.kernel
def set_color_by_material(mat_color: ti.types.ndarray()):
    for i in range(n_particles):
        mat = F_materials[i]
        if mat < mat_color.shape[0]:
            F_colors[i] = ti.Vector(
                [mat_color[mat, 0], mat_color[mat, 1], mat_color[mat, 2], 1.0]
            )
        else:
            F_colors[i] = ti.Vector([1.0, 1.0, 1.0, 1.0])


print("Loading scenes...this might take a minute")


scene_dir = CURRENT_DIR / "scenes"
scene_entries: List["SceneEntry"] = []
scene_loader_messages: List[str] = []

if load_scenes is not None:
    definitions = load_scenes(scene_dir)
    if not definitions:
        scene_loader_messages.append("No scenes found in scenes/.")
    for definition in definitions:
        entry = _build_scene_entry_from_definition(definition)
        if entry is None:
            label = getattr(definition, "title", getattr(definition, "key", str(definition)))
            scene_loader_messages.append(f"Scene '{label}' skipped: no usable objects.")
            continue
        scene_entries.append(entry)
        for warning in entry.warnings:
            scene_loader_messages.append(f"[{entry.title}] {warning}")
    if scene_entries:
        scene_loader_messages.append(f"Loaded {len(scene_entries)} scene(s).")
else:
    scene_loader_messages.append("Scene loader unavailable; cannot load scenes.")

scene_names = [entry.title for entry in scene_entries]
curr_scene_id = 0

paused = False

use_random_colors = False
show_world_grid = False
show_simulation_grid = False
particles_radius = DEFAULT_PARTICLE_RADIUS

show_object_gizmos = False
pivot_bbox_vertices: Optional[Any] = None
pivot_axis_vertices: Dict[str, Optional[Any]] = {"x": None, "y": None, "z": None}
pivot_bbox_vertex_count = 0
pivot_axis_vertex_counts: Dict[str, int] = {"x": 0, "y": 0, "z": 0}

# Simulation timeline state
DEFAULT_TIMELINE_DURATION = 10.0
timeline_duration = DEFAULT_TIMELINE_DURATION
simulation_time = 0.0
sim_frame_count = 0
sim_substep_count = 0
timeline_auto_pause = False

material_colors = [
    (0.1, 0.6, 0.9),
    (0.93, 0.33, 0.23),
    (1.0, 1.0, 1.0),
    (0.55, 0.55, 0.55),
]


def reset_timeline():
    global simulation_time, sim_frame_count, sim_substep_count
    simulation_time = 0.0
    sim_frame_count = 0
    sim_substep_count = 0


def apply_scene_overrides(scene_entry: SceneEntry):
    global timeline_duration
    timeline_duration = DEFAULT_TIMELINE_DURATION
    GRAVITY[:] = DEFAULT_GRAVITY
    overrides = scene_entry.overrides
    gravity_override = overrides.get("gravity") if isinstance(overrides, dict) else None
    if gravity_override is not None:
        for i in range(3):
            GRAVITY[i] = float(gravity_override[i])
    timeline_override = overrides.get("timeline_duration") if isinstance(overrides, dict) else None
    if timeline_override is not None:
        try:
            timeline_duration = float(timeline_override)
        except (TypeError, ValueError):
            pass


def init():
    global paused
    reset_timeline()
    if not scene_entries:
        print("[mpm3dExt] No scenes available to initialize.")
        return
    current_scene = scene_entries[curr_scene_id]
    init_vols(current_scene.shapes)
    if not use_random_colors:
        set_color_by_material(np.array(material_colors, dtype=np.float32))


def load_scene(index: int):
    global curr_scene_id, pivot_bbox_vertices, pivot_axis_vertices
    global pivot_bbox_vertex_count, pivot_axis_vertex_counts
    if not scene_entries:
        pivot_bbox_vertices = None
        pivot_axis_vertices = {"x": None, "y": None, "z": None}
        pivot_bbox_vertex_count = 0
        pivot_axis_vertex_counts = {"x": 0, "y": 0, "z": 0}
        return
    index = max(0, min(index, len(scene_entries) - 1))
    curr_scene_id = index
    apply_scene_overrides(scene_entries[curr_scene_id])
    rebuild_pivot_debug_geometry()
    init()


load_scene(curr_scene_id)


res = (1080, 720)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)

canvas = window.get_canvas()
gui = window.get_gui()
scene = window.get_scene()
camera_start_position = tuple(domain_scalar(v) for v in (0.5, 1.0, 1.95))
camera_start_lookat = tuple(domain_scalar(v) for v in (0.5, 0.3, 0.5))
camera = ti.ui.Camera()
camera.position(*camera_start_position)
camera.lookat(*camera_start_lookat)
camera.fov(55)


class OrbitCameraController:
    def __init__(self, camera):
        self.camera = camera
        self.center = np.array(camera.curr_lookat, dtype=np.float32)
        offset = np.array(camera.curr_position, dtype=np.float32) - self.center
        self.distance = max(float(np.linalg.norm(offset)), 1e-3)
        self.azimuth = float(np.arctan2(offset[0], offset[2]))
        ratio = offset[1] / self.distance if self.distance > 1e-6 else 0.0
        self.elevation = float(np.arcsin(np.clip(ratio, -0.999, 0.999)))
        self.rotation_speed = 2.0 * math.pi
        self.pan_speed = 1.5
        self.zoom_speed = 2.5
        self.keyboard_speed = 2.0
        self.min_distance = max(domain_scalar(0.1), 1e-3)
        self.max_distance = domain_scalar(20.0)
        self.drag_mode = None
        self.last_cursor = None
        self.last_time = time.perf_counter()

    @staticmethod
    def _normalize(vec):
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return np.zeros(3, dtype=np.float32)
        return vec / norm

    def _compute_position(self):
        cos_elev = math.cos(self.elevation)
        return self.center + np.array(
            [
                self.distance * math.sin(self.azimuth) * cos_elev,
                self.distance * math.sin(self.elevation),
                self.distance * math.cos(self.azimuth) * cos_elev,
            ],
            dtype=np.float32,
        )

    def update(self, window):
        now = time.perf_counter()
        dt = max(now - self.last_time, 1e-6)
        self.last_time = now

        cursor = np.array(window.get_cursor_pos(), dtype=np.float32)
        position = self._compute_position()
        forward = self._normalize(self.center - position)
        if not np.any(forward):
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        right = self._normalize(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        if not np.any(right):
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = self._normalize(np.cross(right, forward))
        if not np.any(up):
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        move = np.zeros(3, dtype=np.float32)
        if window.is_pressed("w"):
            move += forward
        if window.is_pressed("s"):
            move -= forward
        if window.is_pressed("a"):
            move -= right
        if window.is_pressed("d"):
            move += right
        if window.is_pressed("e"):
            move += up
        if window.is_pressed("q"):
            move -= up
        if np.linalg.norm(move) > 1e-6:
            move = self._normalize(move)
            self.center += move * self.keyboard_speed * dt * max(self.distance, domain_scalar(0.1))

        mode = None
        if window.is_pressed(ti.ui.RMB):
            if window.is_pressed(ti.ui.CTRL):
                mode = "zoom"
            elif window.is_pressed(ti.ui.SHIFT):
                mode = "pan"
            else:
                mode = "orbit"
        elif window.is_pressed(ti.ui.MMB):
            mode = "pan"

        if mode is None:
            self.drag_mode = None
            self.last_cursor = cursor
        else:
            if self.drag_mode != mode or self.last_cursor is None:
                self.drag_mode = mode
                self.last_cursor = cursor
            dx = cursor[0] - self.last_cursor[0]
            dy = cursor[1] - self.last_cursor[1]
            if mode == "orbit":
                self.azimuth -= dx * self.rotation_speed
                self.elevation += dy * self.rotation_speed
                limit = math.pi * 0.49
                self.elevation = float(np.clip(self.elevation, -limit, limit))
            elif mode == "pan":
                pan_scale = self.pan_speed * self.distance
                self.center += (-right * dx + up * dy) * pan_scale
            elif mode == "zoom":
                zoom_scale = math.exp(dy * self.zoom_speed)
                self.distance = float(np.clip(self.distance * zoom_scale, self.min_distance, self.max_distance))
            self.last_cursor = cursor

        self.distance = float(np.clip(self.distance, self.min_distance, self.max_distance))

        new_position = self._compute_position()
        new_forward = self._normalize(self.center - new_position)
        if not np.any(new_forward):
            new_forward = forward
        new_right = self._normalize(np.cross(new_forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        if not np.any(new_right):
            new_right = right
        new_up = self._normalize(np.cross(new_right, new_forward))
        if not np.any(new_up):
            new_up = up

        self.camera.position(*new_position)
        self.camera.lookat(*self.center)
        self.camera.up(*new_up)

camera_controller = OrbitCameraController(camera)

def show_options():
    global use_random_colors
    global paused
    global particles_radius
    global curr_scene_id
    global show_world_grid
    global show_simulation_grid
    global show_object_gizmos
    global timeline_duration
    global timeline_auto_pause
    global simulation_time
    global sim_frame_count
    global sim_substep_count

    with gui.sub_window("Scenes", 0.05, 0.1, 0.26, 0.24) as w:
        if not scene_entries:
            w.text("No scenes available")
        else:
            old_scene = curr_scene_id
            for i, name in enumerate(scene_names):
                if w.checkbox(name, curr_scene_id == i):
                    curr_scene_id = i
            if curr_scene_id != old_scene:
                load_scene(curr_scene_id)
                paused = True

            active_scene = scene_entries[curr_scene_id]
            w.text(f"source: {active_scene.source}")
            if active_scene.description:
                for line in active_scene.description.splitlines():
                    w.text(line[:64])
            w.text(f"objects: {len(active_scene.shapes)}")
            max_listed = 3
            for summary in active_scene.object_summaries[:max_listed]:
                w.text(f"- {summary}")
            remaining = len(active_scene.object_summaries) - max_listed
            if remaining > 0:
                w.text(f"- (+{remaining} more)")
            overrides = active_scene.overrides if isinstance(active_scene.overrides, dict) else {}
            gravity_override = overrides.get("gravity")
            if gravity_override is not None:
                w.text(
                    "gravity: "
                    f"({gravity_override[0]:.2f}, {gravity_override[1]:.2f}, {gravity_override[2]:.2f})"
                )
            if overrides.get("timeline_duration") is not None:
                w.text(f"timeline default: {overrides['timeline_duration']:.2f} s")
            if active_scene.warnings:
                w.text("Scene warnings:")
                lim = 2
                for warning in active_scene.warnings[:lim]:
                    w.text(warning[:64])
                extra_warnings = len(active_scene.warnings) - lim
                if extra_warnings > 0:
                    w.text(f"... (+{extra_warnings})")
        if scene_loader_messages:
            w.text("Messages:")
            max_msgs = 3
            for msg in scene_loader_messages[:max_msgs]:
                w.text(msg[:64])
            extra = len(scene_loader_messages) - max_msgs
            if extra > 0:
                w.text(f"... (+{extra})")

    with gui.sub_window("Gravity", 0.05, 0.36, 0.26, 0.12) as w:
        GRAVITY[0] = w.slider_float("x", GRAVITY[0], -10, 10)
        GRAVITY[1] = w.slider_float("y", GRAVITY[1], -10, 10)
        GRAVITY[2] = w.slider_float("z", GRAVITY[2], -10, 10)

    with gui.sub_window("Timeline", 0.34, 0.1, 0.34, 0.24) as w:
        status_text = "paused" if paused else "running"
        w.text(f"status: {status_text}")
        w.text(f"time: {simulation_time:.3f} s")
        timeline_duration = w.slider_float("duration (s)", timeline_duration, 0.0, 120.0)
        if timeline_duration > 0.0:
            progress_fraction = min(simulation_time / timeline_duration, 1.0)
            remaining = max(timeline_duration - simulation_time, 0.0)
            w.text(f"remaining: {remaining:.3f} s")
        else:
            progress_fraction = 0.0
            w.text("remaining: --")
        bar_length = 24
        filled = min(bar_length, max(0, int(round(progress_fraction * bar_length))))
        bar = "#" * filled + "-" * (bar_length - filled)
        w.text(f"progress: [{bar}] {progress_fraction * 100:.1f}%")
        if timeline_duration > 0.0 and simulation_time >= timeline_duration:
            status_suffix = " (auto paused)" if timeline_auto_pause and paused else ""
            w.text(f"timeline reached target{status_suffix}")
        sim_fps = sim_frame_count / simulation_time if simulation_time > 1e-6 else 0.0
        substep_rate = sim_substep_count / simulation_time if simulation_time > 1e-6 else 0.0
        w.text(f"frames: {sim_frame_count} ({sim_fps:.1f} fps)")
        w.text(f"substeps: {sim_substep_count} ({substep_rate:.1f}/s)")
        timeline_auto_pause = w.checkbox("auto pause at end", timeline_auto_pause)
        if w.button("reset timeline"):
            reset_timeline()

    with gui.sub_window("Options", 0.05, 0.5, 0.26, 0.35) as w:
        use_random_colors = w.checkbox("use_random_colors", use_random_colors)
        if not use_random_colors:
            material_colors[WATER] = w.color_edit_3("water color", material_colors[WATER])
            material_colors[SNOW] = w.color_edit_3("snow color", material_colors[SNOW])
            material_colors[JELLY] = w.color_edit_3("jelly color", material_colors[JELLY])
            material_colors[CONCRETE] = w.color_edit_3(
                "concrete color", material_colors[CONCRETE]
            )
            set_color_by_material(np.array(material_colors, dtype=np.float32))
        particles_radius = w.slider_float("particles radius (m)", particles_radius, 0, MAX_PARTICLE_RADIUS)
        if w.button("restart"):
            init()
        if paused:
            if w.button("Continue"):
                paused = False
        else:
            if w.button("Pause"):
                paused = True

    with gui.sub_window("World Grid", 0.75, 0.45, 0.2, 0.22) as w:
        show_world_grid = w.checkbox("show world grid", show_world_grid)
        show_simulation_grid = w.checkbox("show simulation grid", show_simulation_grid)
        show_object_gizmos = w.checkbox("show object gizmos", show_object_gizmos)
        w.text(f"domain: 0-{SIMULATION_LENGTH:.2f} m")
        w.text(f"grid: {n_grid}^3 cells")
        w.text(f"dx: {dx:.4e} m")
        if show_world_grid:
            tick_preview = [f"{value:.1f}" for value in world_grid_tick_labels]
            if tick_preview:
                midpoint = (len(tick_preview) + 1) // 2
                w.text("ticks (m):")
                w.text(", ".join(tick_preview[:midpoint]))
                if midpoint < len(tick_preview):
                    w.text(", ".join(tick_preview[midpoint:]))
        if show_object_gizmos:
            if pivot_bbox_vertex_count > 0:
                w.text(f"object gizmo edges: {pivot_bbox_vertex_count // 2}")
            else:
                w.text("object gizmos: none")
        if show_simulation_grid and simulation_grid_line_count:
            w.text(f"sim lines: {simulation_grid_line_count}")
        if (show_world_grid or show_simulation_grid) and axis_label_line_count:
            w.text(f"label strokes: {axis_label_line_count}")

def render():
    camera_controller.update(window)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    colors_used = F_colors_random if use_random_colors else F_colors
    scene.particles(F_x, per_vertex_color=colors_used, radius=particles_radius)

    scene.point_light(pos=tuple(domain_scalar(v) for v in (0.5, 1.5, 0.5)), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=tuple(domain_scalar(v) for v in (0.5, 1.5, 1.5)), color=(0.5, 0.5, 0.5))

    if show_world_grid:
        if world_grid_vertices is not None and world_grid_vertices.shape[0] > 0:
            scene.lines(world_grid_vertices, width=1.2, color=(0.6, 0.6, 0.6))
        scene.lines(axis_x_vertices, width=2.0, color=(0.9, 0.3, 0.3))
        scene.lines(axis_y_vertices, width=2.0, color=(0.3, 0.8, 0.3))
        scene.lines(axis_z_vertices, width=2.0, color=(0.3, 0.4, 0.9))

    if show_object_gizmos:
        if pivot_bbox_vertices is not None and pivot_bbox_vertex_count > 0:
            scene.lines(pivot_bbox_vertices, width=1.0, color=PIVOT_BBOX_COLOR)
        if isinstance(pivot_axis_vertices, dict):
            for axis_key, color in PIVOT_AXIS_COLORS.items():
                verts_field = pivot_axis_vertices.get(axis_key)
                if verts_field is not None and pivot_axis_vertex_counts.get(axis_key, 0) > 0:
                    scene.lines(verts_field, width=1.4, color=color)

    if show_simulation_grid and simulation_grid_vertices is not None and simulation_grid_vertices.shape[0] > 0:
        scene.lines(simulation_grid_vertices, width=0.8, color=(0.35, 0.45, 0.75))

    if (show_world_grid or show_simulation_grid) and axis_label_vertices is not None and axis_label_vertices.shape[0] > 0:
        scene.lines(axis_label_vertices, width=1.1, color=AXIS_LABEL_COLOR)

    canvas.scene(scene)


def main():
    global simulation_time, sim_frame_count, sim_substep_count, paused, timeline_auto_pause
    frame_id = 0

    while window.running:
        # print("heyyy ",frame_id)
        frame_id += 1
        frame_id = frame_id % 256

        if not paused:
            for _ in range(steps):
                substep(*GRAVITY)
            simulation_time += steps * dt
            sim_substep_count += steps
            sim_frame_count += 1
            if timeline_auto_pause and timeline_duration > 0.0 and simulation_time >= timeline_duration:
                simulation_time = min(simulation_time, timeline_duration)
                paused = True

        render()
        show_options()
        window.show()


if __name__ == "__main__":
    main()


