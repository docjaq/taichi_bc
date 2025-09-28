"""Utilities for loading JSON-based MPM scene descriptions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _load_scene_file(path: Path) -> Optional[SceneDefinition]:
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
            objects.append(
                SceneObject(
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
            )
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

        objects.append(
            SceneObject(
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
        )

    return SceneDefinition(
        key=key,
        title=title,
        description=description,
        path=path,
        objects=objects,
        simulation_overrides=overrides,
        warnings=warnings,
    )


def load_scenes(scene_dir: Path) -> List[SceneDefinition]:
    scenes: List[SceneDefinition] = []
    if not scene_dir.exists() or not scene_dir.is_dir():
        return scenes

    for path in sorted(scene_dir.glob(f"*{SCENE_FILE_SUFFIX}")):
        scene = _load_scene_file(path)
        if scene is None:
            continue
        scenes.append(scene)
    return scenes
