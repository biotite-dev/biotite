# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite"
__author__ = "Patrick Kunzmann"
__all__ = ["colors", "plot_scaled_text", "set_font_size_in_coord", "AdaptiveFancyArrow"]

import warnings
from collections import OrderedDict
import numpy as np
from numpy.linalg import norm

_FONT_PROPERTY_KEYS = ["family", "style", "variant", "weight", "stretch", "size"]


# Biotite themed colors
colors = OrderedDict(
    [
        ("brightorange", "#ffb569ff"),
        ("lightorange", "#ff982dff"),
        ("orange", "#ff8405ff"),
        ("dimorange", "#dc7000ff"),
        ("darkorange", "#b45c00ff"),
        ("brightgreen", "#98e97fff"),
        ("lightgreen", "#6fe04cff"),
        ("green", "#52da2aff"),
        ("dimgreen", "#45bc20ff"),
        ("darkgreen", "#389a1aff"),
    ]
)


def plot_scaled_text(
    axes, text, x, y, width=None, height=None, mode="unlocked", **kwargs
):
    """
    Create a :class:`matplotlib.textpath.TextPath`, whose text size is given in
    coordinates instead of font size.

    Parameters
    ----------
    axes : Axes
        The axes to draw the text on.
    text : str
        The text to be drawn.
    x, y : float
        The lower left position of the text.
    width, height : float, optional
        The width/height `text` should have in its reference coordinate system.
        At least one value must be supplied.
    mode : {'proportional', 'unlocked', 'maximum', 'minimum'}, optional
        Defines how the text size is scaled.
        The scaling mode:

            - *proportional* - The width and height are scaled by the
              same extent.
              Either `width` or `height` must be set for this mode.
            - *unlocked* - The width and the height are scaled by
              different extents, changing the aspect ratio of the text.
              Both `width` and `height` must be set for this mode.
            - *maximum* - The width and the height are scaled by
              the same extent, so that they are at maximum as large
              as the supplied `width`/`height`.
              Both `width` and `height` must be set for this mode.
            - *minimum* - The width and the height are scaled by
              the same extent, so that they are at minimum as large
              as the supplied `width`/`height`.
              Both `width` and `height` must be set for this mode.

    **kwargs
        Additional parameters for the :class:`matplotlib.font_manager.FontProperties`
        of the text or the created :class:`matplotlib.patches.PathPatch`.

    Returns
    -------
    patch : matplotlib.patches.PathPatch
        The patch that represents the text.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    # The larger the size, the more there is an offset at the x-axis
    # -> Keep font size small to reduce this error to a minimum
    # The size is transformed to the desired size afterwards anyway
    font_property_kwargs = {
        key: kwargs.pop(key) for key in _FONT_PROPERTY_KEYS if key in kwargs
    }
    path = TextPath((x, y), text, size=1e-3, prop=font_property_kwargs)
    bbox = path.get_extents()

    if mode == "proportional":
        if width is None:
            # Proportional scaling based on height
            scale_y = height / bbox.height
            scale_x = scale_y
        elif height is None:
            # Proportional scaling based on width
            scale_x = width / bbox.width
            scale_y = scale_x
        else:
            raise ValueError(
                "width or height are mutually exclusive in 'proportional' mode"
            )
    elif mode == "unlocked":
        scale_x = width / bbox.width
        scale_y = height / bbox.height
    elif mode == "minimum":
        scale_x = width / bbox.width
        scale_y = height / bbox.height
        scale = max(scale_x, scale_y)
        scale_x, scale_y = scale, scale
    elif mode == "maximum":
        scale_x = width / bbox.width
        scale_y = height / bbox.height
        scale = min(scale_x, scale_y)
        scale_x, scale_y = scale, scale

    path = (
        Affine2D()
        .translate(-bbox.x0, -bbox.y0)
        .scale(scale_x, scale_y)
        .translate(bbox.x0, bbox.y0)
        .transform_path(path)
    )
    patch = PathPatch(path, **kwargs)
    axes.add_patch(patch)

    return patch


def set_font_size_in_coord(text, width=None, height=None, mode="unlocked"):
    """
    Specifiy the font size of an existing `Text` object in coordinates
    of the object's reference coordiante system.

    Instead of having the font size fixed in 'pt', the size of the text
    scales to the specied width/height and adapts to changes in the
    plot's width/height.
    The scaling can be proportional or non-proportional, depending
    the `mode`.

    DEPRECATED: Use :func:`plot_scaled_text()` instead.

    Parameters
    ----------
    text : Text:
        The matplotlib `Text` to be scaled.
    width, height : float, optional
        The new width/height of `text` in its
        reference coordinate system.
        At least one value must be supplied.
    mode : {'proportional', 'unlocked', 'maximum', 'minimum'}, optional
        The scaling mode:

            - *proportional* - The width and height are scaled by the
              same extent.
              Either `width` or `height` must be set for this mode.
            - *unlocked* - The width and the height are scaled by
              different extents, changing the aspect ratio.
              Both `width` and `height` must be set for this mode.
            - *maximum* - The width and the height are scaled by
              the same extent, so that they are at maximum as large
              as the supplied `width`/`height`.
              Both `width` and `height` must be set for this mode.
            - *minimum* - The width and the height are scaled by
              the same extent, so that they are at minimum as large
              as the supplied `width`/`height`.
              Both `width` and `height` must be set for this mode.

    Notes
    -----
    This function uses the :func:`get_window_extent()` method of the
    :class:`Text` object.
    According to experience, this function does not give the the exact
    visual boundaries of the text.
    Consequently, the scaled text might be slightly smaller or larger
    than the specified width/height.
    This behavior is not equal for all initial font sizes (in 'pt'),
    the boundaries for an initial size of 1 'pt' seem to be most exact.
    """
    from matplotlib.patheffects import AbstractPathEffect
    from matplotlib.transforms import Affine2D, Bbox

    class TextScaler(AbstractPathEffect):
        def __init__(self, text, width, height, mode):
            self._text = text
            self._mode = mode
            self._width = width
            self._height = height

        def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):  # noqa: N803
            ax = self._text.axes
            try:
                renderer = ax.get_figure().canvas.get_renderer()
            except Exception:
                # Use cached renderer for backends, where
                # `get_renderer()` is not available
                # Based on the strategy from `Text.get_window_extent()`
                renderer = ax.get_figure()._cachedRenderer
            if renderer is None:
                raise
            bbox = text.get_window_extent(renderer)
            bbox = Bbox(ax.transData.inverted().transform(bbox))

            if self._mode == "proportional":
                if self._width is None:
                    # Proportional scaling based on height
                    scale_y = self._height / bbox.height
                    scale_x = scale_y
                elif self._height is None:
                    # Proportional scaling based on width
                    scale_x = self._width / bbox.width
                    scale_y = scale_x
            elif self._mode == "unlocked":
                scale_x = self._width / bbox.width
                scale_y = self._height / bbox.height
            elif self._mode == "minimum":
                scale_x = self._width / bbox.width
                scale_y = self._height / bbox.height
                scale = max(scale_x, scale_y)
                scale_x, scale_y = scale, scale
            elif self._mode == "maximum":
                scale_x = self._width / bbox.width
                scale_y = self._height / bbox.height
                scale = min(scale_x, scale_y)
                scale_x, scale_y = scale, scale

            affine = Affine2D().scale(scale_x, scale_y) + affine
            renderer.draw_path(gc, tpath, affine, rgbFace)

    warnings.warn(
        "Deprecated, use 'biotite.graphics.text.plot_scaled_text()' instead.",
        DeprecationWarning,
    )

    if mode in ["unlocked", "minimum", "maximum"]:
        if width is None or height is None:
            raise TypeError(f"Width and height must be set in '{mode}' mode")
    elif mode == "proportional":
        if not (width is None and height is not None) or not (
            height is None and width is not None
        ):
            raise TypeError(f"Either width or height must be set in '{mode}' mode")
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    text.set_path_effects([TextScaler(text, width, height, mode)])


try:
    # Only create this class when matplotlib is installed
    from matplotlib.patches import FancyArrow
    from matplotlib.transforms import Bbox

    class AdaptiveFancyArrow(FancyArrow):
        """
        A `FancyArrow` with fixed head shape.

        The length of the head is proportional to the width of the head
        in display coordinates.
        If the head length is longer than the length of the entire
        arrow, the head length is limited to the arrow length.

        Parameters
        ----------
        x,y  : float
            The arrow's start position.
        dx, dy : float
            The arrow's direction vector, inclduing the arrow head.
        tail_width, head_width : float
            The width of the arrow's tail and head in its reference
            coordinate system
        head_ratio : float, optional
            The length of the arrow head as faction of the arrow width
            (display coordinates).
        draw_head: bool, optional
            If false, the arrow has no head. The result is a rectangle.
        shape : str, optional
            The `shape` parameter in the constructor of `FancyArrow`.
        **kwargs
            Other parameters that are used in the constructor of
            `FancyArrow`.
        """

        def __init__(
            self,
            x,
            y,
            dx,
            dy,
            tail_width,
            head_width,
            head_ratio,
            draw_head=True,
            shape="full",
            **kwargs,
        ):
            self._x = x
            self._y = y
            self._dx = dx
            self._dy = dy
            self._tail_width = tail_width
            self._head_width = head_width
            self._head_ratio = head_ratio
            self._draw_head = draw_head
            self._shape = shape
            self._kwargs = kwargs
            if not draw_head:
                head_width = tail_width
            super().__init__(
                x,
                y,
                dx,
                dy,
                width=tail_width,
                head_width=head_width,
                overhang=0,
                shape=shape,
                length_includes_head=True,
                **kwargs,
            )

        def draw(self, renderer):
            arrow_box = Bbox([(0, 0), (0, self._head_width)])
            arrow_box_display = self.axes.transData.transform_bbox(arrow_box)
            head_length_display = np.abs(arrow_box_display.height * self._head_ratio)
            arrow_box_display.x1 = arrow_box_display.x0 + head_length_display
            # Transfrom back to data coordinates for plotting
            arrow_box = self.axes.transData.inverted().transform_bbox(arrow_box_display)
            head_length = arrow_box.width
            arrow_length = norm((self._dx, self._dy))
            if head_length > arrow_length:
                # If the head would be longer than the entire arrow,
                # only draw the arrow head with reduced length
                head_length = arrow_length
            if not self._draw_head:
                head_length = 0

            # Renew the arrow's properties
            super().__init__(
                self._x,
                self._y,
                self._dx,
                self._dy,
                width=self._tail_width,
                head_width=self._head_width,
                overhang=0,
                shape=self._shape,
                head_length=head_length,
                length_includes_head=True,
                axes=self.axes,
                transform=self.get_transform(),
                **self._kwargs,
            )
            self.set_clip_path(self.axes.patch)
            super().draw(renderer)

        # Override to replace docstring
        # Removes warning:
        # unknown document: /tutorials/intermediate/constrainedlayout_guide
        def get_in_layout(self):
            """ """
            return super().get_in_layout()

        def set_in_layout(self, in_layout):
            """ """
            return super().set_in_layout(in_layout)

except ImportError:
    # Dummy class that propagates a meaningful error,
    # i.e. that Matplotlib is not installed
    class AdaptiveFancyArrow:
        def __init__(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'matplotlib'")
