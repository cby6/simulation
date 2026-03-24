import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

MU0 = 4 * np.pi * 1e-7

# CAD layout: 24 toroidal stations; at each station, 12 coils on one circle (288 total).
# The circle lies in a poloidal plane (its axis / plane normal is the toroidal tangent).
N_TOROIDAL = 24
N_COILS_PER_CIRCLE = 12


# =========================
# Geometry helpers
# =========================
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_ring_coils(
    n_coils,
    ring_radius,
    z0,
    axis_mode="tangent",
    phase_deg=0.0,
    m_mag=1.0
):
    """
    Create one planar circular ring of dipole coils (xy plane, optional z offset).

    For the full toroidal CAD layout (24×12), use build_coil_array() instead.

    Parameters
    ----------
    n_coils : int
        Number of coils in this ring.
    ring_radius : float
        Radius of this coil ring.
    z0 : float
        z coordinate of this ring.
    axis_mode : str
        'tangent', 'radial', or 'z'
    phase_deg : float
        Angular offset in degrees.
    m_mag : float
        Dipole moment magnitude for each coil.

    Returns
    -------
    coils : list of dict
        Each dict has:
          center : np.array([x,y,z])
          axis   : np.array([mx,my,mz]) unit direction
          m_mag  : float
          phi    : float
    """
    coils = []
    phase = np.deg2rad(phase_deg)

    for k in range(n_coils):
        phi = phase + 2 * np.pi * k / n_coils
        c = np.array([ring_radius * np.cos(phi), ring_radius * np.sin(phi), z0])

        e_r = np.array([np.cos(phi), np.sin(phi), 0.0])
        e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])
        e_z = np.array([0.0, 0.0, 1.0])

        if axis_mode == "tangent":
            axis = e_phi
        elif axis_mode == "radial":
            axis = e_r
        elif axis_mode == "z":
            axis = e_z
        else:
            raise ValueError(f"Unknown axis_mode: {axis_mode}")

        coils.append(
            {
                "center": c,
                "axis": unit(axis),
                "m_mag": float(m_mag),
                "phi": phi,
            }
        )
    return coils


def build_coil_array(
    r_major=0.95,
    r_coil_circle=0.12,
    m_mag=1.0,
    poloidal_phase_deg=0.0,
    toroidal_phase_deg=0.0,
):
    """
    Build the full toroidal coil set: N_TOROIDAL × N_COILS_PER_CIRCLE = 288 dipoles.

    At each toroidal angle, N_COILS_PER_CIRCLE coils lie on a circle in the poloidal
    plane (normal to that plane = toroidal tangent e_phi), so the circle's axis is
    tangential to the torus. Each dipole moment is along the poloidal tangent to that
    small circle (direction of travel around the 12-coil ring).

    Parameters
    ----------
    r_major : float
        Major radius of the torus centerline (m).
    r_coil_circle : float
        Radius of the circle on which the 12 coil centers lie in each poloidal plane (m).
    m_mag : float
        Dipole moment magnitude per coil.
    poloidal_phase_deg : float
        Rotates the 12-coil pattern within each poloidal plane.
    toroidal_phase_deg : float
        Rotates the 24 toroidal stations around the device.
    """
    coils = []
    poloidal_phase = np.deg2rad(poloidal_phase_deg)
    toroidal_phase = np.deg2rad(toroidal_phase_deg)

    for k in range(N_TOROIDAL):
        phi_t = toroidal_phase + 2 * np.pi * k / N_TOROIDAL
        e_r = np.array([np.cos(phi_t), np.sin(phi_t), 0.0])
        e_z = np.array([0.0, 0.0, 1.0])

        for j in range(N_COILS_PER_CIRCLE):
            alpha = poloidal_phase + 2 * np.pi * j / N_COILS_PER_CIRCLE
            # Circle in poloidal plane (normal = toroidal tangent); centers follow minor circle.
            center = r_major * e_r + r_coil_circle * (
                np.cos(alpha) * e_r + np.sin(alpha) * e_z
            )
            # Poloidal tangent along the 12-coil circle (solenoid axis direction).
            t_pol = -np.sin(alpha) * e_r + np.cos(alpha) * e_z
            coils.append(
                {
                    "center": center,
                    "axis": unit(t_pol),
                    "m_mag": float(m_mag),
                    "phi": phi_t,
                    "poloidal": alpha,
                    "toroidal_idx": k,
                    "poloidal_idx": j,
                }
            )
    return coils


# =========================
# Dipole magnetic field
# =========================
def dipole_B(r_obs, r_src, m_vec, eps=1e-9):
    """
    Magnetic field of a point dipole.
    r_obs, r_src, m_vec are 3-vectors.
    """
    R = r_obs - r_src
    R2 = np.dot(R, R) + eps
    Rmag = np.sqrt(R2)
    Rhat = R / Rmag

    factor = MU0 / (4 * np.pi * Rmag**3)
    return factor * (3 * np.dot(m_vec, Rhat) * Rhat - m_vec)


def total_B_at_point(r_obs, coils, states):
    """
    Sum dipole fields from all ON coils.
    states[i] in {-1, 0, +1}
      +1 means coil ON with +moment direction
      -1 means coil ON with reversed moment
       0 means OFF
    """
    B = np.zeros(3)
    for coil, s in zip(coils, states):
        if s == 0:
            continue
        m_vec = s * coil["m_mag"] * coil["axis"]
        B += dipole_B(r_obs, coil["center"], m_vec)
    return B


# =========================
# Activation patterns
# =========================
def traveling_window_sequence(n_coils, window=6, reverse=False):
    """
    One moving ON-window around the array.
    Returns a list of states arrays.
    """
    seq = []
    for start in range(n_coils):
        states = np.zeros(n_coils, dtype=int)
        for j in range(window):
            idx = (start + j) % n_coils
            states[idx] = 1
        seq.append(states[::-1] if reverse else states)
    return seq


def three_phase_sequence(n_coils, phase_step=1):
    """
    Example 3-phase style switching:
    group A = +1, group B = -1, group C = 0,
    then rotate the grouping.
    """
    seq = []
    labels = np.array([0, 1, 2] * ((n_coils + 2) // 3))[:n_coils]

    for shift in range(0, n_coils, phase_step):
        rolled = np.roll(labels, shift)
        states = np.zeros(n_coils, dtype=int)
        states[rolled == 0] = 1
        states[rolled == 1] = -1
        states[rolled == 2] = 0
        seq.append(states)
    return seq


def custom_order_sequence(n_coils, order, pulse_len=1):
    """
    Turn on one or more coils following a custom order.
    order is a list of coil indices.
    """
    seq = []
    for idx in order:
        states = np.zeros(n_coils, dtype=int)
        for j in range(pulse_len):
            states[(idx + j) % n_coils] = 1
        seq.append(states)
    return seq


def rotating_triangle_toroidal_sequence(n_toroidal, n_per_array, rotate_step=1):
    """
    Activate 3 coils at a time in one 12-coil array, 120 degrees apart.

    Frame t:
      - active toroidal array: k = t % n_toroidal
      - triangle rotates in that array by rotate_step each frame:
            j0 = (t * rotate_step) % n_per_array
            j in {j0, j0 + n_per_array/3, j0 + 2*n_per_array/3}

    For the default 24x12 layout, the first frames are:
      t=0 -> [0, 4, 8]
      t=1 -> [13, 17, 21]
      t=2 -> [26, 30, 34]
    """
    if n_per_array % 3 != 0:
        raise ValueError("n_per_array must be divisible by 3 for 120-degree spacing.")

    total = n_toroidal * n_per_array
    sep = n_per_array // 3
    seq = []

    for t in range(n_toroidal):
        states = np.zeros(total, dtype=int)
        k = t % n_toroidal
        j0 = (t * rotate_step) % n_per_array
        for m in range(3):
            j = (j0 + m * sep) % n_per_array
            idx = k * n_per_array + j
            states[idx] = 1
        seq.append(states)
    return seq


# =========================
# Visualization
# =========================
def coil_xy_positions(coils):
    xy = np.array([c["center"][:2] for c in coils])
    return xy[:, 0], xy[:, 1]


def animate_center_field(
    coils,
    sequence,
    save_path="torus_center_field.gif",
    fps=8,
    title="Center magnetic field animation"
):
    """
    Animate active coils and center magnetic field history.
    """
    n_coils = len(coils)
    center = np.array([0.0, 0.0, 0.0])

    B_hist = []
    for states in sequence:
        Bc = total_B_at_point(center, coils, states)
        B_hist.append(Bc)
    B_hist = np.array(B_hist)

    Bmag = np.linalg.norm(B_hist, axis=1)
    x, y = coil_xy_positions(coils)

    # Figure layout
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, 4)

    # Left: top-view coil layout
    ax0.set_aspect("equal")
    ax0.set_title("Top view: coil states")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    # Big torus guide circle
    radii = np.linalg.norm(np.array([c["center"][:2] for c in coils]), axis=1)
    Rmean = np.mean(radii)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax0.plot(Rmean * np.cos(theta), Rmean * np.sin(theta), "--", alpha=0.4)

    scatter = ax0.scatter(x, y, s=80, c="lightgray", edgecolors="k")
    center_point = ax0.scatter([0], [0], s=120, c="black", marker="x")

    # field arrow at center in x-y plane
    arrow_scale = max(np.max(np.abs(B_hist[:, :2])), 1e-12)
    q = ax0.quiver(
        [0], [0],
        [0], [0],
        angles="xy",
        scale_units="xy",
        scale=arrow_scale * 4,
        color="blue",
        width=0.008,
    )

    lim = 1.25 * np.max(np.sqrt(x**2 + y**2))
    ax0.set_xlim(-lim, lim)
    ax0.set_ylim(-lim, lim)

    # Right-top: Bx, By, Bz vs frame
    ax1.set_title("Center field components")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("B at center (T)")
    line_bx, = ax1.plot([], [], label="Bx")
    line_by, = ax1.plot([], [], label="By")
    line_bz, = ax1.plot([], [], label="Bz")
    vline1 = ax1.axvline(0, color="k", ls="--", alpha=0.5)
    ax1.legend()

    ax1.set_xlim(0, len(sequence) - 1)
    comp_max = np.max(np.abs(B_hist))
    if comp_max < 1e-15:
        comp_max = 1.0
    ax1.set_ylim(-1.1 * comp_max, 1.1 * comp_max)

    # Right-bottom: |B|
    ax2.set_title("Center field magnitude")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("|B| (T)")
    line_mag, = ax2.plot([], [])
    vline2 = ax2.axvline(0, color="k", ls="--", alpha=0.5)
    ax2.set_xlim(0, len(sequence) - 1)
    mag_max = np.max(Bmag)
    if mag_max < 1e-15:
        mag_max = 1.0
    ax2.set_ylim(0, 1.1 * mag_max)

    txt = fig.text(0.52, 0.92, "", fontsize=11)

    def colors_from_states(states):
        colors = []
        for s in states:
            if s > 0:
                colors.append("tomato")
            elif s < 0:
                colors.append("gold")
            else:
                colors.append("lightgray")
        return colors

    def update(frame):
        states = sequence[frame]
        Bc = B_hist[frame]

        scatter.set_color(colors_from_states(states))

        # Update center arrow using Bx, By only
        q.set_UVC([Bc[0]], [Bc[1]])

        line_bx.set_data(np.arange(frame + 1), B_hist[:frame + 1, 0])
        line_by.set_data(np.arange(frame + 1), B_hist[:frame + 1, 1])
        line_bz.set_data(np.arange(frame + 1), B_hist[:frame + 1, 2])
        line_mag.set_data(np.arange(frame + 1), Bmag[:frame + 1])

        vline1.set_xdata([frame, frame])
        vline2.set_xdata([frame, frame])

        txt.set_text(
            f"{title}\n"
            f"frame={frame}   "
            f"B_center = [{Bc[0]:.3e}, {Bc[1]:.3e}, {Bc[2]:.3e}] T   "
            f"|B| = {np.linalg.norm(Bc):.3e} T"
        )

        return scatter, q, line_bx, line_by, line_bz, line_mag, vline1, vline2, txt

    ani = FuncAnimation(fig, update, frames=len(sequence), interval=1000 / fps, blit=False)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved animation to: {save_path}")


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    coils = build_coil_array()
    n = len(coils)

    # Pick ONE of these sequences:

    # 1) Traveling 3-coil triangle:
    # one 12-coil array active at a time, with indices separated by 4.
    sequence = rotating_triangle_toroidal_sequence(
        n_toroidal=N_TOROIDAL,
        n_per_array=N_COILS_PER_CIRCLE,
        rotate_step=1,
    )

    # 2) Moving ON-window (legacy)
    # sequence = traveling_window_sequence(n_coils=n, window=8)

    # 3) 3-phase rotating pattern
    # sequence = three_phase_sequence(n_coils=n, phase_step=1)

    # 4) Custom coil firing order
    # order = [0, 3, 6, 9, 12, 15, 18, 21]
    # sequence = custom_order_sequence(n_coils=n, order=order, pulse_len=2)

    animate_center_field(
        coils=coils,
        sequence=sequence,
        save_path="torus_center_field.gif",
        fps=2,
        title="Magnetic field at torus center"
    )
