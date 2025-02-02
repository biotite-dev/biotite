import shutil
from os.path import dirname, join, splitext
from IPython.display import Image
from sphinx_gallery.py_source_parser import extract_file_config
from sphinx_gallery.scrapers import figure_rst

STATIC_IMAGE_COMMAND = "static_image"


def static_image_scraper(block, block_vars, gallery_conf):
    script_dir = dirname(block_vars["src_file"])
    image_sources = []
    _, code, _ = block
    # Search for `sphinx_gallery_static_image` commands
    block_conf = extract_file_config(code)
    if STATIC_IMAGE_COMMAND not in block_conf:
        return ""

    image_sources = [
        join(script_dir, image_name.strip())
        for image_name in block_conf[STATIC_IMAGE_COMMAND].split(",")
    ]

    # Copy the images into the 'gallery' directory under a canonical
    # sphinx-gallery name
    image_destinations = []
    image_path_iterator = block_vars["image_path_iterator"]
    for image in image_sources:
        suffix = splitext(image)[1]
        image_destination = image_path_iterator.next()
        # Replace destination file suffix with the suffix from the input
        # file, e.g. '.png' with '.gif' for animated images
        image_destination = splitext(image_destination)[0] + suffix
        image_destinations.append(image_destination)
        shutil.copy(image, image_destination)

    # Generate rST for detected image files
    return figure_rst(image_destinations, gallery_conf["src_dir"])


def pymol_scraper(block, block_vars, gallery_conf):
    image_path_iterator = block_vars["image_path_iterator"]
    image_paths = []

    for object in block_vars["example_globals"].values():
        if isinstance(object, Image) and object.metadata["source"] == "PyMOL":
            image_path = next(image_path_iterator)
            with open(image_path, "wb") as image_file:
                image_file.write(object.data)
            image_paths.append(image_path)
    return figure_rst(
        image_paths,
        gallery_conf["src_dir"],
    )
