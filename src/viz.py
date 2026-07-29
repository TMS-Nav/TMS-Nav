# shared rendering, one plotter setup so the coil demo and the target preview
# frame the head the same way
import numpy as np
import pyvista as pv


def _base_plotter(scalp):
    p = pv.Plotter(off_screen=True, window_size=(1600, 1200))

    # otherwise coil edges and marker outlines come out jagged
    p.enable_anti_aliasing("ssaa")

    p.add_mesh(scalp, color="lightgray", smooth_shading=True, specular=0.15)
    return p


def _add_axes(p):
    # ticks in mm so the scale is checkable
    p.show_grid(
        xtitle="x  R+ [mm]",
        ytitle="y  A+ [mm]",
        ztitle="z  S+ [mm]",
        n_xlabels=3,
        n_ylabels=3,
        n_zlabels=3,
        fmt="%.0f",
        font_size=16,
        location="outer",
        padding=0.02,
    )

    # corner triad, tells us which way the head is facing
    p.add_axes(
        xlabel="R",
        ylabel="A",
        zlabel="S",
        line_width=5,
        labels_off=False,
        viewport=(0.0, 0.0, 0.22, 0.22),
    )


def _face_camera(p, scalp, view_dir):
    # sit the camera out along view_dir looking back at the head
    view = np.asarray(view_dir, dtype=float)
    view = view / np.linalg.norm(view)

    focus = np.array(scalp.center)
    p.camera_position = [tuple(focus + view * 500.0), tuple(focus), (0.0, 0.0, 1.0)]

    # keeps the direction, pulls back until the head fits
    p.reset_camera()
    p.camera.zoom(0.95)


def render_coil(scalp, coil, contact, n, gap, out_path):
    """Scalp plus one seated figure-8 coil, saved to out_path."""

    p = _base_plotter(scalp)
    p.add_mesh(coil, color="#1f77b4", smooth_shading=True, specular=0.5, specular_power=20)

    _add_axes(p)
    p.add_text(
        f"figure-8 coil, {gap:.2f} mm off scalp\n"
        f"contact RAS ({contact[0]:.0f}, {contact[1]:.0f}, {contact[2]:.0f}) mm",
        font_size=14,
    )

    # off the coil normal toward the front, shows both wings and the face
    _face_camera(p, scalp, np.asarray(n, dtype=float) + np.array([0.0, 0.9, 0.0]))

    p.screenshot(str(out_path))
    return out_path


def render_targets(scalp, targets, out_path, title="scalp targets"):
    """Scalp plus a labeled marker at each target, saved to out_path."""

    p = _base_plotter(scalp)

    label_pts = []
    labels = []
    for t in targets:
        # ball perched on the scalp point, nudged out a bit so it sits on top
        center = t.contact + t.normal * 3.0
        p.add_mesh(pv.Sphere(radius=6.0, center=center), color=t.color)
        label_pts.append(center)
        labels.append(t.label)

    p.add_point_labels(
        np.array(label_pts),
        labels,
        font_size=18,
        text_color="black",
        shape_opacity=0.25,
        point_size=1,
        always_visible=True,
    )

    _add_axes(p)
    p.add_text(title, font_size=14)

    # from the front and a bit up, all three sit on the front/top of the head
    _face_camera(p, scalp, np.array([0.0, 1.0, 0.6]))

    p.screenshot(str(out_path))
    return out_path
