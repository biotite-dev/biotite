from glob import glob
import shutil
import copy
import sys
import os
from os.path import splitext, join, dirname, isfile
from sphinx_gallery.scrapers import figure_rst
from sphinx.errors import ExtensionError


def static_image_scraper(block, block_vars, gallery_conf):
    script_dir = dirname(block_vars["src_file"])
    image_sources = []
    # Search for comment line containing 'biotite_static_image'
    _, code, _ = block
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("#"):
            line = line[1:].strip()
            if line.startswith("biotite_static_image"):
                # Get the image name after the '=' character
                image_name = line.split("=")[1].strip()
                image_sources.append(join(script_dir, image_name))

    # Copy the images into the 'gallery' directory under a canonical
    # sphinx-gallery name
    image_destinations = []
    image_path_iterator = block_vars['image_path_iterator']
    for image in image_sources:
        suffix = splitext(image)[1]
        image_destination = image_path_iterator.next()
        # Replace destination file suffix with the suffix from the input
        # file, e.g. '.png' with '.gif' for animated images
        image_destination = splitext(image_destination)[0] + suffix
        image_destinations.append(image_destination)
        shutil.copy(image, image_destination)
    
    # Generate rST for detected image files
    return figure_rst(image_destinations, gallery_conf['src_dir'])


def pymol_scraper(block, block_vars, gallery_conf):
    # Search for comment line containing 'Visualization with PyMOL...'
    _, code, _ = block
    if any([
        line.strip() == "# Visualization with PyMOL..."
        for line in code.splitlines()
    ]):
        pymol_script_path = splitext(block_vars["src_file"])[0] + "_pymol.py"
        pymol_image_path  = splitext(block_vars["src_file"])[0] + ".png"
        if not isfile(pymol_script_path):
            raise ExtensionError(
                f"'{block_vars['src_file']}' has no corresponding "
                f"'{pymol_script_path}' file"
            )
        
        # If PyMOL image is already created, do not run PyMOL script,
        # ad this should not be required for building the documentation
        if not isfile(pymol_image_path):
            # Create a shallow copy,
            # to avoid ading new variables to example script
            script_globals = copy.copy(block_vars["example_globals"])
            script_globals["__image_destination__"] = pymol_image_path
            
            try:
                import pymol
            except ImportError:
                raise ExtensionError("PyMOL is not installed")
            try:
                import ammolite
            except ImportError:
                raise ExtensionError("Ammolite is not installed")     
            with open(pymol_script_path, "r") as script:
                # Prevent PyMOL from writing stuff (splash screen, etc.)
                # to STDOUT or STDERR
                # -> Save original STDOUT/STDERR and point them
                # temporarily to DEVNULL
                dev_null = open(os.devnull, 'w')
                orig_stdout = sys.stdout
                orig_stderr = sys.stderr
                sys.stdout = dev_null
                sys.stderr = dev_null
                try:
                    exec(script.read(), script_globals)
                except Exception as e:
                    raise ExtensionError(
                        f"PyMOL script raised a {type(e).__name__}: {str(e)}"
                    )
                finally:
                    # Restore STDOUT/STDERR
                    sys.stdout = orig_stdout
                    sys.stderr = orig_stderr
                    dev_null.close()
            if not isfile(pymol_image_path):
                raise ExtensionError(
                    "PyMOL script did not create an image "
                    "(at expected location)"
                )

        # Copy the images into the 'gallery' directory under a canonical
        # sphinx-gallery name
        image_path_iterator = block_vars['image_path_iterator']
        image_destination = image_path_iterator.next()
        shutil.copy(pymol_image_path, image_destination)
        return figure_rst([image_destination], gallery_conf['src_dir'])
    
    else:
        return figure_rst([], gallery_conf['src_dir'])
