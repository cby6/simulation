import numpy as np
from manim import (
    Arrow3D,
    BLUE,
    DEGREES,
    GREY_B,
    GREY_D,
    ORIGIN,
    RED,
    Torus,
    ThreeDScene,
    VGroup,
    ValueTracker,
)

from simulation.traveling_magnetic_field import (
    N_COILS_PER_CIRCLE,
    N_TOROIDAL,
    build_coil_array,
    rotating_triangle_toroidal_sequence,
    total_B_at_point,
)


def _state_color(s: int):
    if s > 0:
        return RED
    if s < 0:
        return "#D4AF37"  # gold-ish
    return GREY_D


class ToroidalCoilFieldScene(ThreeDScene):
    def construct(self):
        # ---- Geometry (match build_coil_array defaults unless overridden here)
        r_major = 0.95
        r_coil_circle = 0.12

        coils = build_coil_array(r_major=r_major, r_coil_circle=r_coil_circle, m_mag=1.0)
        n = len(coils)

        # ---- Switching sequence (traveling 3-coil triangle)
        sequence = rotating_triangle_toroidal_sequence(
            n_toroidal=N_TOROIDAL,
            n_per_array=N_COILS_PER_CIRCLE,
            rotate_step=1,
        )

        # ---- Precompute center field (for stable scaling)
        center = np.array([0.0, 0.0, 0.0])
        B_hist = np.array([total_B_at_point(center, coils, states) for states in sequence])
        Bmag = np.linalg.norm(B_hist, axis=1)
        b_scale = float(np.max(Bmag)) if float(np.max(Bmag)) > 0 else 1.0

        # ---- Camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=35 * DEGREES, zoom=2.5)

        # ---- Torus guide
        torus = Torus(
            major_radius=r_major,
            minor_radius=0.06,
            resolution=(36, 18),
        ).set_color(GREY_B)
        torus.set_opacity(0.35)
        self.add(torus)

        # ---- Coil axes as short 3D arrows (fast + readable)
        # Length picked to be visible relative to r_coil_circle.
        coil_axis_len = 0.08
        coil_mobs = []
        for coil in coils:
            c = coil["center"]
            a = coil["axis"]
            start = c - 0.5 * coil_axis_len * a
            end = c + 0.5 * coil_axis_len * a
            mob = Arrow3D(
                start=start,
                end=end,
                thickness=0.01,
                height=0.08,
                base_radius=0.03,
                resolution=8,
                color=GREY_D,
            )
            mob.set_opacity(0.85)
            coil_mobs.append(mob)

        coils_group = VGroup(*coil_mobs)
        self.add(coils_group)

        # ---- Center field arrow
        arrow_len = 0.5  # visual length in scene units
        field_arrow = Arrow3D(
            start=ORIGIN,
            end=np.array([arrow_len, 0.0, 0.0]),
            thickness=0.02,
            height=0.12,
            base_radius=0.05,
            resolution=12,
            color=BLUE,
        )
        self.add(field_arrow)

        # ---- Updaters driven by a frame index tracker
        frame = ValueTracker(0.0)

        def _frame_index():
            # clamp to [0, len(sequence)-1]
            idx = int(np.clip(np.round(frame.get_value()), 0, len(sequence) - 1))
            return idx

        def update_coils(_group):
            idx = _frame_index()
            states = sequence[idx]
            for mob, s in zip(coil_mobs, states):
                mob.set_color(_state_color(int(s)))
                mob.set_opacity(0.25 if int(s) == 0 else 0.95)
            return _group

        def update_field_arrow(mob):
            idx = _frame_index()
            b = B_hist[idx]
            b_norm = np.linalg.norm(b)
            direction = b / b_norm if b_norm > 0 else np.array([1.0, 0.0, 0.0])
            end = arrow_len * (b_norm / b_scale) * direction
            # Recreate Arrow3D for correct 3D geometry update.
            new = Arrow3D(
                start=ORIGIN,
                end=end,
                thickness=0.02,
                height=0.12,
                base_radius=0.05,
                resolution=12,
                color=BLUE,
            )
            mob.become(new)
            return mob

        coils_group.add_updater(update_coils)
        field_arrow.add_updater(update_field_arrow)

        # ---- Animate through all frames
        self.play(frame.animate.set_value(len(sequence) - 1), run_time=10, rate_func=lambda t: t)
        self.wait(0.5)

        # Cleanup updaters
        coils_group.remove_updater(update_coils)
        field_arrow.remove_updater(update_field_arrow)

