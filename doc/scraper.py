from glob import glob
import shutil
import os
from sphinx_gallery.scrapers import figure_rst



def static_scraper(block, block_vars, gallery_conf):
    script_path = os.path.dirname(block_vars["src_file"])
    image_sources = []
    _, code, _ = block
    lines = code.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            line = line[1:].strip()
            if line.startswith("biotite_static_image"):
                image_name = line.split("=")[1].strip()
                image_sources.append(os.path.join(script_path, image_name))

    image_destinations = []
    image_path_iterator = block_vars['image_path_iterator']
    for image in image_sources:
        image_destination = image_path_iterator.next()
        image_destinations.append(image_destination)
        shutil.copy(image, image_destination)
    # Use the `figure_rst` helper function to generate rST for image files
    return figure_rst(image_destinations, gallery_conf['src_dir'])