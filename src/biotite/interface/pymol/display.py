__name__ = "biotite.interface.pymol"
__author__ = "Patrick Kunzmann"
__all__ = ["TimeoutError", "RenderError", "show", "play"]

import datetime
import shutil
import subprocess
import tempfile
import time
from os import remove
from os.path import getsize, join
from biotite.interface.pymol.startup import get_and_set_pymol_instance

_INTERVAL = 0.1


class TimeoutError(Exception):
    """
    Exception that is raised after time limit expiry in :func:`show()`.
    """

    pass


class RenderError(Exception):
    """
    Exception that is raised when ``imagemagick`` or ``ffmpeg`` fails.
    """

    pass


def show(size=None, use_ray=False, timeout=60.0, pymol_instance=None):
    """
    Render an image of the *PyMOL* session and display it in the current
    *Jupyter* notebook.

    Note that this function works only in a *Jupyter* notebook.

    Parameters
    ----------
    size : tuple of (int, int), optional
        The width and height of the rendered image in pixels.
        By default, the size of the current *PyMOL* viewport is used.
    use_ray : bool, optional
        If set to true, the a ray-traced image is created.
        This will also increase the rendering time.
    timeout : float
        The number of seconds to wait for image output from *PyMOL*.
    pymol_instance : module or SingletonPyMOL or PyMOL, optional
        If *PyMOL* is used in library mode, the :class:`PyMOL`
        or :class:`SingletonPyMOL` object is given here.
        If otherwise *PyMOL* is used in GUI mode, the :mod:`pymol`
        module is given.
        By default the currently active *PyMOL* instance is used.
        If no *PyMOL* instance is currently running,
        *PyMOL* is started in library mode.

    Raises
    ------
    TimeoutError
        If no image was created after expiry of the `timeout` limit.

    Notes
    -----
    Internally uses the *PyMOL* ``png`` command.
    """
    try:
        from IPython.display import Image
    except ImportError:
        raise ImportError("IPython is not installed")

    pymol_instance = get_and_set_pymol_instance(pymol_instance)
    cmd = pymol_instance.cmd

    if size is None:
        width = 0
        height = 0
    else:
        width, height = size

    image_file = tempfile.NamedTemporaryFile(
        delete=False, prefix="biotite_", suffix=".png"
    )
    # Close directly and cleanup later
    # This is necessary, as Windows does not allow writing opened files
    image_file.close()

    start_time = datetime.datetime.now()
    cmd.png(image_file.name, width, height, ray=int(use_ray))
    while True:
        # After 'timeout' seconds the loop exits with an error
        if (datetime.datetime.now() - start_time).total_seconds() > timeout:
            raise TimeoutError("No PNG image was output within the expected time limit")
        # Check if PyMOL has already written image data to file
        if getsize(image_file.name) > 0:
            break
        time.sleep(_INTERVAL)

    with open(image_file.name, "rb") as f:
        image_data = f.read()
    remove(image_file.name)
    return Image(image_data, embed=True, metadata={"source": "PyMOL"})


def play(
    size=None, fps=30, format="gif", html_attributes="controls", pymol_instance=None
):
    """
    Render an video of the *PyMOL* video frames and display it in the current
    *Jupyter* notebook.

    Note that this function works only in a *Jupyter* notebook.

    Parameters
    ----------
    size : tuple of (int, int), optional
        The width and height of the rendered video in pixels.
        By default, the size of the current *PyMOL* viewport is used.
    fps : int
        The number of frames per second.
    format : {"gif", "mp4"}, optional
        The format of the rendered video.
        By default, a GIF is created.
    html_attributes : str, optional
        The HTML attributes that are passed to the ``<video>`` tag.
        Only used, if ``format="mp4"``.
    pymol_instance : module or SingletonPyMOL or PyMOL, optional
        If *PyMOL* is used in library mode, the :class:`PyMOL`
        or :class:`SingletonPyMOL` object is given here.
        If otherwise *PyMOL* is used in GUI mode, the :mod:`pymol`
        module is given.
        By default the currently active *PyMOL* instance is used.
        If no *PyMOL* instance is currently running,
        *PyMOL* is started in library mode.

    Notes
    -----
    Internally uses the *PyMOL* ``mpng`` command.
    This function requires either the ``ffmpeg`` (``gif`` or ``mp4``)
    or ``imagemagick`` (``gif``) command line tool to be installed.
    """
    try:
        from IPython.display import Image, Video
    except ImportError:
        raise ImportError("IPython is not installed")

    pymol_instance = get_and_set_pymol_instance(pymol_instance)
    cmd = pymol_instance.cmd

    if size is None:
        width = 0
        height = 0
    else:
        width, height = size

    with tempfile.TemporaryDirectory(prefix="biotite_") as frame_dir:
        # Must use ray tracing, as no window is created
        # Otherwise PyMOL raises 'MoviePNG-Error: Missing rendered image.'
        cmd.mpng(join(frame_dir, "img_"), mode=2, width=width, height=height)
        video_data = _create_video(frame_dir, fps, format)

    if format == "mp4":
        return Video(
            video_data,
            embed=True,
            mimetype="video/mp4",
            html_attributes=html_attributes,
        )
    else:
        return Image(video_data, embed=True, metadata={"source": "PyMOL"})


def _create_video(input_dir, fps, format):
    """
    Create a video from the images in the given directory using ``ffmpeg```
    or ``imagemagick``.

    Parameters
    ----------
    input_dir : str
        The directory containing the frames to be concatenated into a video.
        The images are consumed in lexographical order.
    fps : int
        The number of frames per second.
    format : {"gif", "mp4"}
        The format of the video.

    Returns
    -------
    video : bytes
        The video data.
    """
    if format == "gif":
        # GIFs created with 'imagemagick' have less artifacts
        if _is_installed("magick"):
            return _create_gif_with_imagemagick(input_dir, fps)
        elif _is_installed("ffmpeg"):
            return _create_video_with_ffmpeg(input_dir, fps, format)
        else:
            raise RenderError("Neither 'imagemagick' nor 'ffmpeg' is installed")
    elif format == "mp4":
        if _is_installed("ffmpeg"):
            return _create_video_with_ffmpeg(input_dir, fps, format)
        else:
            raise RenderError("'ffmpeg' is not installed")


def _create_gif_with_imagemagick(input_dir, fps):
    # See https://usage.imagemagick.org/anim_basics/ for reference
    completed_process = subprocess.run(
        [
            "magick",
            # GIFs require a multiple of a hundredth of a second to work properly
            "-delay", str(int(100 / fps)),
            # Make animation loop infinitely
            "-loop", "0",
            # Do not overlay images
            "-dispose", "Previous",
            join(input_dir, "*.png"),
            # Decrease GIF size
            "-layers", "Optimize",
            "GIF:-",
        ],
        capture_output=True,
    )  # fmt: skip
    if completed_process.returncode != 0:
        raise RenderError(completed_process.stderr.decode())
    return completed_process.stdout


def _create_video_with_ffmpeg(input_dir, fps, format):
    # 'input_dir' is a temporary directory anyway
    video_path = join(input_dir, f"video.{format}")
    completed_process = subprocess.run(
        [
            "ffmpeg",
            "-i", join(input_dir, "img_%04d.png"),
            "-r", str(fps),
            # Must be set to obtain a non-corrupted video
            "-pix_fmt", "yuv420p",
            video_path,
        ],
        capture_output=True,
    )  # fmt: skip
    if completed_process.returncode != 0:
        raise RenderError(completed_process.stderr.decode())
    with open(video_path, "rb") as f:
        video_data = f.read()
    return video_data


def _is_installed(program):
    """
    Check whether the given program is installed.

    Parameters
    ----------
    program : str
        The name of the program to check.

    Returns
    -------
    installed : bool
        True, if the program is installed, false otherwise.
    """
    return shutil.which(program) is not None
