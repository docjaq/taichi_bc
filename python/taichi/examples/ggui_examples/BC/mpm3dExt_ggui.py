import math
import time

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)

# Simulation configuration (SI units)
DIM = 3
GRID_RESOLUTION = 96  # cells per edge; increase with SIMULATION_LENGTH to keep resolution
SUBSTEPS_PER_FRAME = 20
MAX_SUBSTEP_DT = 1.2e-4
SIMULATION_LENGTH = 1.0  # metres, edge length of the cubic domain
PACKING_FRACTION = 0.5  # fraction of cell size used for initial particle spacing
BASE_DENSITY = 1000.0  # kg/m^3
BASE_YOUNG_MODULUS = 1.0e5  # Pa
POISSON_RATIO = 0.2
GRAVITY = [0.0, -9.81, 0.0]  # m/s^2
BOUNDARY_CELLS = 3
CFL_NUMBER = 0.5

WATER_STIFFNESS_MULTIPLIER = 1.5
WATER_DAMPING_PER_SECOND = 0.999
JELLY_HARDENING = 0.35
JELLY_STIFFNESS_MULTIPLIER = 0.85
JELLY_DAMPING_PER_SECOND = 0.9
SNOW_STIFFNESS_MULTIPLIER = 2.2
SNOW_DAMPING_PER_SECOND = 0.995
CONCRETE_HARDENING_MULTIPLIER = 12.0
CONCRETE_STIFFNESS_MULTIPLIER = 4.0
CONCRETE_DAMPING_PER_SECOND = 0.98
CONCRETE_MIN_PLASTIC_STRETCH = 1.0 - 1.5e-3
CONCRETE_MAX_PLASTIC_STRETCH = 1.0 + 1.5e-3

DEFAULT_PARTICLE_RADIUS = 0.02 * SIMULATION_LENGTH
MAX_PARTICLE_RADIUS = 0.1 * SIMULATION_LENGTH

dim = DIM
n_grid = GRID_RESOLUTION

dx = SIMULATION_LENGTH / n_grid

n_particles = n_grid**dim // 2 ** (dim - 1)
print(f"Particles: {n_particles}")
print(f"dx = {dx:.4e} m")

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
    def __init__(self, minimum, size, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material


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
):
    for i in range(first_par, last_par):
        F_x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector([x_size, y_size, z_size]) + ti.Vector(
            [x_begin, y_begin, z_begin]
        )
        F_Jp[i] = 1
        F_dg[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_v[i] = ti.Vector([0.0, 0.0, 0.0])
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


def init_vols(vols):
    set_all_unused()
    total_vol = 0
    for v in vols:
        total_vol += v.volume

    next_p = 0
    for i, v in enumerate(vols):
        v = vols[i]
        if isinstance(v, CubeVolume):
            par_count = int(v.volume / total_vol * n_particles)
            if i == len(vols) - 1:  # this is the last volume, so use all remaining particles
                par_count = n_particles - next_p
            init_cube_vol(next_p, next_p + par_count, *v.minimum, *v.size, v.material)
            next_p += par_count
        else:
            raise Exception("???")


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


print("Loading presets...this might take a minute")

presets = [
    [
        CubeVolume(domain_vector([0.55, 0.05, 0.55]), domain_vector([0.4, 0.4, 0.4]), WATER),
    ],
    [
        CubeVolume(domain_vector([0.05, 0.05, 0.05]), domain_vector([0.3, 0.4, 0.3]), WATER),
        CubeVolume(domain_vector([0.65, 0.05, 0.65]), domain_vector([0.3, 0.4, 0.3]), WATER),
    ],
    [
        CubeVolume(domain_vector([0.6, 0.05, 0.6]), domain_vector([0.25, 0.25, 0.25]), WATER),
        CubeVolume(domain_vector([0.35, 0.35, 0.35]), domain_vector([0.25, 0.25, 0.25]), SNOW),
        CubeVolume(domain_vector([0.05, 0.6, 0.05]), domain_vector([0.25, 0.25, 0.25]), JELLY),
    ],
    [
        CubeVolume(domain_vector([0.6, 0.05, 0.6]), domain_vector([0.25, 0.25, 0.25]), JELLY),
        #CubeVolume(domain_vector([0.35, 0.35, 0.35]), domain_vector([0.25, 0.25, 0.25]), SNOW),
        CubeVolume(domain_vector([0.55, 0.8, 0.55]), domain_vector([0.15, 0.15, 0.15]), CONCRETE),
        CubeVolume(domain_vector([0.75, 0.8, 0.75]), domain_vector([0.05, 0.15, 0.05]), CONCRETE),
        CubeVolume(domain_vector([0.3, 0.8, 0.3]), domain_vector([0.15, 0.15, 0.15]), CONCRETE),
    ],
]
preset_names = [
    "Single Dam Break",
    "Double Dam Break",
    "Water Snow Jelly",
    "Jelly Snow Concrete",
]

curr_preset_id = 0

paused = False

use_random_colors = False
particles_radius = DEFAULT_PARTICLE_RADIUS

material_colors = [
    (0.1, 0.6, 0.9),
    (0.93, 0.33, 0.23),
    (1.0, 1.0, 1.0),
    (0.55, 0.55, 0.55),
]


def init():
    global paused
    init_vols(presets[curr_preset_id])
    if not use_random_colors:
        set_color_by_material(np.array(material_colors, dtype=np.float32))


init()

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
    global curr_preset_id

    with gui.sub_window("Presets", 0.05, 0.1, 0.2, 0.15) as w:
        old_preset = curr_preset_id
        for i in range(len(presets)):
            if w.checkbox(preset_names[i], curr_preset_id == i):
                curr_preset_id = i
        if curr_preset_id != old_preset:
            init()
            paused = True

    with gui.sub_window("Gravity", 0.05, 0.3, 0.2, 0.1) as w:
        GRAVITY[0] = w.slider_float("x", GRAVITY[0], -10, 10)
        GRAVITY[1] = w.slider_float("y", GRAVITY[1], -10, 10)
        GRAVITY[2] = w.slider_float("z", GRAVITY[2], -10, 10)

    with gui.sub_window("Options", 0.05, 0.45, 0.2, 0.4) as w:
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


def render():
    camera_controller.update(window)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    colors_used = F_colors_random if use_random_colors else F_colors
    scene.particles(F_x, per_vertex_color=colors_used, radius=particles_radius)

    scene.point_light(pos=tuple(domain_scalar(v) for v in (0.5, 1.5, 0.5)), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=tuple(domain_scalar(v) for v in (0.5, 1.5, 1.5)), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)


def main():
    frame_id = 0

    while window.running:
        # print("heyyy ",frame_id)
        frame_id += 1
        frame_id = frame_id % 256

        if not paused:
            for _ in range(steps):
                substep(*GRAVITY)

        render()
        show_options()
        window.show()


if __name__ == "__main__":
    main()
