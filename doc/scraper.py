from glob import glob
import shutil
import os
from os.path import splitext
from sphinx_gallery.scrapers import figure_rst


def static_image_scraper(block, block_vars, gallery_conf):
    script_path = os.path.dirname(block_vars["src_file"])
    image_sources = []
    # Search for comment line containing 'biotite_static_image'
    _, code, _ = block
    lines = code.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            line = line[1:].strip()
            if line.startswith("biotite_static_image"):
                # Get the image name after the '=' character
                image_name = line.split("=")[1].strip()
                image_sources.append(os.path.join(script_path, image_name))

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