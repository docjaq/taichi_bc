"""Utilities for loading JSON-based MPM scene descriptions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

SCENE_FILE_SUFFIX = ".mpm.json"
DEFAULT_PIVOT = (0.5, 0.0, 0.5)


@dataclass
class SceneObject:
    scene_id: str
    obj_type: str
    material: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    rotation_euler: Tuple[float, float, float]
    initial_velocity: Tuple[float, float, float]
    pivot: Tuple[float, float, float]
    color_override: Optional[Tuple[float, float, float]] = None
    center: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None


@dataclass
class SceneDefinition:
    key: str
    title: str
    description: str
    path: Optional[Path]
    objects: List[SceneObject]
    simulation_overrides: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def short_summary(self) -> str:
        return f"{len(self.objects)} object(s)"


def _to_float(value: object, name: str, warnings: List[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        warnings.append(f"'{name}' must be a number")
        return None
    return number


def _to_vec3(
    values: Sequence[object],
    name: str,
    warnings: List[str],
    default: Tuple[float, float, float],
    allow_missing: bool = False,
) -> Tuple[float, float, float]:
    if values is None:
        if allow_missing:
            return default
        warnings.append(f"missing '{name}', using default {default}")
        return default
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        warnings.append(f"invalid '{name}' type, expected sequence of 3 numbers; using default {default}")
        return default
    if len(values) != 3:
        warnings.append(f"'{name}' expected 3 values, got {len(values)}; using default {default}")
        return default
    try:
        vec = tuple(float(v) for v in values)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        warnings.append(f"'{name}' contains non-numeric values; using default {default}")
        return default
    return vec  # type: ignore[return-value]


def _clamp_relative_vec(vec: Tuple[float, float, float], name: str, warnings: List[str]) -> Tuple[float, float, float]:
    clamped = []
    for i, value in enumerate(vec):
        clamped_value = value
        if value < 0.0 or value > 1.0:
            clamped_value = min(max(value, 0.0), 1.0)
            axis = "xyz"[i] if i < 3 else str(i)
            warnings.append(f"'{name}' component {axis}={value:.3f} clamped to {clamped_value:.3f}")
        clamped.append(clamped_value)
    return tuple(clamped)  # type: ignore[return-value]


AXIS_NAMES = ("x", "y", "z")
Vec3 = Tuple[float, float, float]
Matrix3 = Tuple[Vec3, Vec3, Vec3]

def _mat3_mul(a: Matrix3, b: Matrix3) -> Matrix3:
    rows: List[Vec3] = []
    for i in range(3):
        row = []
        for j in range(3):
            total = 0.0
            for k in range(3):
                total += a[i][k] * b[k][j]
            row.append(total)
        rows.append(tuple(row))
    return tuple(rows)  # type: ignore[return-value]

def _mat3_vec_mul(matrix: Matrix3, vec: Vec3) -> Vec3:
    return tuple(
        sum(matrix[i][j] * vec[j] for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]

def _rotation_matrix_from_euler(angles: Sequence[float]) -> Matrix3:
    if angles is None:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    if len(angles) != 3:
        raise ValueError("rotation_euler must contain exactly 3 values")
    ax, ay, az = (float(a) for a in angles)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx: Matrix3 = (
        (1.0, 0.0, 0.0),
        (0.0, cx, -sx),
        (0.0, sx, cx),
    )
    ry: Matrix3 = (
        (cy, 0.0, sy),
        (0.0, 1.0, 0.0),
        (-sy, 0.0, cy),
    )
    rz: Matrix3 = (
        (cz, -sz, 0.0),
        (sz, cz, 0.0),
        (0.0, 0.0, 1.0),
    )
    return _mat3_mul(_mat3_mul(rz, ry), rx)

def _compute_cube_aabb(obj: SceneObject) -> Optional[Tuple[Vec3, Vec3]]:
    if obj.position is None or obj.size is None:
        return None
    size = obj.size
    pivot = obj.pivot if obj.pivot is not None else DEFAULT_PIVOT
    rotation = obj.rotation_euler if obj.rotation_euler is not None else (0.0, 0.0, 0.0)
    try:
        rotation_matrix = _rotation_matrix_from_euler(rotation)
    except ValueError:
        rotation_matrix = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    pivot_world = tuple(obj.position[i] + size[i] * pivot[i] for i in range(3))
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for u in (0.0, 1.0):
        for v in (0.0, 1.0):
            for w in (0.0, 1.0):
                local = (
                    (u - pivot[0]) * size[0],
                    (v - pivot[1]) * size[1],
                    (w - pivot[2]) * size[2],
                )
                rotated = _mat3_vec_mul(rotation_matrix, local)
                world = tuple(rotated[i] + pivot_world[i] for i in range(3))
                for axis in range(3):
                    value = world[axis]
                    if value < mins[axis]:
                        mins[axis] = value
                    if value > maxs[axis]:
                        maxs[axis] = value
    return (tuple(mins), tuple(maxs))  # type: ignore[return-value]

def _compute_sphere_aabb(obj: SceneObject) -> Optional[Tuple[Vec3, Vec3]]:
    center = obj.center
    if center is None and obj.position is not None and obj.size is not None:
        center = tuple(obj.position[i] + 0.5 * obj.size[i] for i in range(3))
    radius = obj.radius
    if radius is None and obj.size is not None:
        radius = max(obj.size) * 0.5
    if center is None or radius is None:
        return None
    radius = float(radius)
    return (
        tuple(center[i] - radius for i in range(3)),
        tuple(center[i] + radius for i in range(3)),
    )  # type: ignore[return-value]

def _compute_scene_object_aabb(obj: SceneObject) -> Optional[Tuple[Vec3, Vec3]]:
    if obj.obj_type == "cube_volume":
        return _compute_cube_aabb(obj)
    if obj.obj_type == "sphere":
        return _compute_sphere_aabb(obj)
    return None

def _ensure_vec3(values: Sequence[object], name: str) -> Vec3:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != 3
    ):
        raise ValueError(f"{name} must be a sequence of three numeric values")
    try:
        return tuple(float(values[i]) for i in range(3))  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

def _coerce_margin(margin: Union[float, Sequence[object]]) -> Vec3:
    if isinstance(margin, Sequence) and not isinstance(margin, (str, bytes)):
        if len(margin) != 3:
            raise ValueError("safety_margin must provide exactly three values")
        values = []
        for index, value in enumerate(margin):
            try:
                scalar = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("safety_margin contains non-numeric values") from exc
            values.append(max(0.0, scalar))
        return tuple(values)  # type: ignore[return-value]
    try:
        scalar = max(0.0, float(margin))
    except (TypeError, ValueError) as exc:
        raise ValueError("safety_margin must be numeric") from exc
    return (scalar, scalar, scalar)

def _validate_scene_object_bounds(
    obj: SceneObject,
    domain_min: Vec3,
    domain_max: Vec3,
    margin: Vec3,
    tolerance: float,
    warning_sink: List[str],
) -> None:
    bounds = _compute_scene_object_aabb(obj)
    if bounds is None:
        return
    aabb_min, aabb_max = bounds
    violations = []
    for axis in range(3):
        min_allowed = domain_min[axis] + margin[axis]
        max_allowed = domain_max[axis] - margin[axis]
        if min_allowed > max_allowed:
            raise ValueError(
                f"invalid bounds configuration: margin on axis {AXIS_NAMES[axis]} leaves no usable space"
            )
        if aabb_min[axis] < min_allowed - tolerance:
            diff = min_allowed - aabb_min[axis]
            violations.append(
                f"-{AXIS_NAMES[axis]} ({aabb_min[axis]:.4f} < {min_allowed:.4f}, by {diff:.4f})"
            )
        if aabb_max[axis] > max_allowed + tolerance:
            diff = aabb_max[axis] - max_allowed
            violations.append(
                f"+{AXIS_NAMES[axis]} ({aabb_max[axis]:.4f} > {max_allowed:.4f}, by {diff:.4f})"
            )
    if violations:
        if any(margin):
            margin_display = tuple(round(m, 5) for m in margin)
            margin_text = f" (margin={margin_display})"
        else:
            margin_text = ""
        warning_sink.append(
            f"object '{obj.scene_id}' ({obj.obj_type}) exceeds domain bounds{margin_text}: "
            + ", ".join(violations)
        )

def _load_scene_file(
    path: Path,
    *,
    domain_min: Vec3,
    domain_max: Vec3,
    margin: Vec3,
    tolerance: float,
) -> Optional[SceneDefinition]:
    warnings: List[str] = []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - best effort logging
        warnings.append(f"failed to parse JSON: {exc}")
        return SceneDefinition(
            key=path.stem,
            title=path.stem,
            description="",
            path=path,
            objects=[],
            warnings=warnings,
        )

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    title = str(meta.get("title", path.stem))
    description = str(meta.get("description", ""))
    key = path.stem

    overrides = payload.get("simulation", {}) if isinstance(payload, dict) else {}
    if not isinstance(overrides, dict):
        warnings.append("'simulation' section must be an object; ignoring overrides")
        overrides = {}

    objects_payload = payload.get("objects", []) if isinstance(payload, dict) else []
    if not isinstance(objects_payload, list):
        warnings.append("'objects' section must be a list; no objects loaded")
        objects_payload = []

    objects: List[SceneObject] = []
    supported_types = {"cube_volume", "sphere"}
    for idx, obj_data in enumerate(objects_payload):
        if not isinstance(obj_data, dict):
            warnings.append(f"object #{idx} is not an object; skipped")
            continue
        raw_type = str(obj_data.get("type", "")).strip()
        if not raw_type:
            warnings.append(f"object #{idx} missing 'type'; skipped")
            continue
        obj_type = raw_type.lower()
        if obj_type not in supported_types:
            warnings.append("object '{}' has unsupported type '{}'; skipped".format(obj_data.get('id', idx), raw_type))
            continue

        material = str(obj_data.get("material", "")).strip().upper()
        if not material:
            warnings.append(f"object '{obj_data.get('id', idx)}' missing 'material'; defaulting to WATER")
            material = "WATER"
        scene_id = str(obj_data.get("id", f"object_{idx}"))

        color_override_payload = obj_data.get("color_override")
        if color_override_payload is not None:
            color_override = _to_vec3(color_override_payload, "color_override", warnings, (0.0, 0.0, 0.0), allow_missing=True)
        else:
            color_override = None

        pivot_raw = _to_vec3(obj_data.get("pivot"), "pivot", warnings, DEFAULT_PIVOT, allow_missing=True)
        pivot = _clamp_relative_vec(pivot_raw, "pivot", warnings)
        velocity = _to_vec3(obj_data.get("initial_velocity"), "initial_velocity", warnings, (0.0, 0.0, 0.0), allow_missing=True)

        if obj_type == "cube_volume":
            position = _to_vec3(obj_data.get("position"), "position", warnings, (0.5, 0.5, 0.5))
            size = _to_vec3(obj_data.get("size"), "size", warnings, (0.2, 0.2, 0.2))
            rotation = _to_vec3(obj_data.get("rotation_euler"), "rotation_euler", warnings, (0.0, 0.0, 0.0), allow_missing=True)
            scene_object = SceneObject(
                scene_id=scene_id,
                obj_type=obj_type,
                material=material,
                position=position,
                size=size,
                rotation_euler=rotation,
                initial_velocity=velocity,
                pivot=pivot,
                color_override=color_override,
            )
            _validate_scene_object_bounds(scene_object, domain_min, domain_max, margin, tolerance, warnings)
            objects.append(scene_object)
            continue

        # Sphere parsing
        center_source = obj_data.get("center")
        center_name = "center"
        if center_source is None:
            center_source = obj_data.get("position")
            center_name = "center" if "center" in obj_data else "position"
        if center_source is None:
            warnings.append(f"sphere '{scene_id}' missing 'center'; skipped")
            continue
        center = _to_vec3(center_source, center_name, warnings, (0.5, 0.5, 0.5), allow_missing=False)
        if "pivot" not in obj_data:
            pivot = (0.5, 0.5, 0.5)

        radius_val = _to_float(obj_data.get("radius"), "radius", warnings)
        if radius_val is None:
            diameter_val = _to_float(obj_data.get("diameter"), "diameter", warnings)
            if diameter_val is not None:
                radius_val = diameter_val * 0.5

        if radius_val is None:
            size_values = obj_data.get("size")
            size_tuple = None
            if isinstance(size_values, Sequence) and not isinstance(size_values, (str, bytes)) and len(size_values) == 3:
                try:
                    size_tuple = tuple(float(v) for v in size_values)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    warnings.append(f"sphere '{scene_id}' has invalid 'size' entries; skipped")
                    size_tuple = None
            if size_tuple is not None:
                if max(size_tuple) - min(size_tuple) > 1e-6:
                    warnings.append(f"sphere '{scene_id}' requires uniform 'size'; values {size_tuple} ignored")
                else:
                    radius_val = size_tuple[0] * 0.5

        if radius_val is None or radius_val <= 0.0:
            warnings.append(f"sphere '{scene_id}' missing a positive radius; skipped")
            continue

        diameter = radius_val * 2.0
        size = (diameter, diameter, diameter)
        position = tuple(center[i] - radius_val for i in range(3))
        rotation = _to_vec3(obj_data.get("rotation_euler"), "rotation_euler", warnings, (0.0, 0.0, 0.0), allow_missing=True)

        scene_object = SceneObject(
            scene_id=scene_id,
            obj_type=obj_type,
            material=material,
            position=position,
            size=size,
            rotation_euler=rotation,
            initial_velocity=velocity,
            pivot=pivot,
            color_override=color_override,
            center=center,
            radius=radius_val,
        )
        _validate_scene_object_bounds(scene_object, domain_min, domain_max, margin, tolerance, warnings)
        objects.append(scene_object)

    return SceneDefinition(
        key=key,
        title=title,
        description=description,
        path=path,
        objects=objects,
        simulation_overrides=overrides,
        warnings=warnings,
    )


def load_scenes(
    scene_dir: Path,
    *,
    domain_min: Sequence[float] = (0.0, 0.0, 0.0),
    domain_max: Sequence[float] = (1.0, 1.0, 1.0),
    safety_margin: Union[float, Sequence[float]] = 0.0,
    tolerance: float = 1e-6,
) -> List[SceneDefinition]:
    scenes: List[SceneDefinition] = []
    if not scene_dir.exists() or not scene_dir.is_dir():
        return scenes

    try:
        domain_min_vec = _ensure_vec3(domain_min, "domain_min")
        domain_max_vec = _ensure_vec3(domain_max, "domain_max")
    except ValueError as exc:
        raise ValueError(f"invalid domain bounds: {exc}") from exc
    try:
        margin_vec = _coerce_margin(safety_margin)
    except ValueError as exc:
        raise ValueError(f"invalid safety_margin: {exc}") from exc
    try:
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance must be numeric") from exc
    if tol < 0.0:
        tol = 0.0
    for axis in range(3):
        if domain_min_vec[axis] > domain_max_vec[axis]:
            raise ValueError(
                f"domain_min[{AXIS_NAMES[axis]}]={domain_min_vec[axis]:.4f} exceeds domain_max[{AXIS_NAMES[axis]}]={domain_max_vec[axis]:.4f}"
            )
        usable = domain_max_vec[axis] - domain_min_vec[axis] - 2.0 * margin_vec[axis]
        if usable < -tol:
            raise ValueError(
                f"safety_margin leaves no usable space along {AXIS_NAMES[axis]} (usable={usable:.4f})"
            )

    for path in sorted(scene_dir.glob(f"*{SCENE_FILE_SUFFIX}")):
        scene = _load_scene_file(
            path,
            domain_min=domain_min_vec,
            domain_max=domain_max_vec,
            margin=margin_vec,
            tolerance=tol,
        )
        if scene is None:
            continue
        scenes.append(scene)
    return scenes







