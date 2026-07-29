# shared rendering, one plotter setup so every preview frames the head the same
# way. draws the scalp plus an optional coil and optional target markers
import numpy as np
import pyvista as pv

COIL_COLOR = "#1f77b4"

# camera directions in RAS, where the eye sits relative to the head
VIEW_DIRS = {
    "front": (0.0, 1.0, 0.5),    # face on, a little up
    "side": (-1.0, 0.2, 0.3),    # from the left, where the coil sits
}


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

    # frame to the head only, otherwise moving the coil around changes how the
    # head sits in the shot
    p.reset_camera(bounds=scalp.bounds)
    p.camera.zoom(0.95)


def _add_markers(p, targets):
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


def render_view(scalp, out_path, coil=None, targets=None, view="front", title=""):
    """Scalp plus optional coil and target markers, from the named view."""

    p = _base_plotter(scalp)

    if coil is not None:
        p.add_mesh(coil, color=COIL_COLOR, smooth_shading=True, specular=0.5, specular_power=20)

    if targets:
        _add_markers(p, targets)

    _add_axes(p)
    if title:
        p.add_text(title, font_size=14)

    _face_camera(p, scalp, VIEW_DIRS.get(view, VIEW_DIRS["front"]))

    p.screenshot(str(out_path))
    return out_path
