import os.path
import os
import codeop
import logging
import copy
from importlib.util import module_from_spec, spec_from_loader
from sphinx.util.logging import getLogger
from sphinx.util import status_iterator
import sphinx_gallery.gen_rst as genrst
from sphinx_gallery.gen_gallery import DEFAULT_GALLERY_CONF
import sphinx_gallery.scrapers as scrapers
import sphinx_gallery.py_source_parser as parser
import biotite


def create_tutorial(src_dir, target_dir):
    logger = getLogger('sphinx-gallery')
    logger.info("generating tutorial...", color="white")
    with open(os.path.join(src_dir, "scripts"), "r") as file:
        scripts = [line.strip() for line in file.read().splitlines()
                   if line[0] != "#" and line.strip() != ""]
    iterator = status_iterator(
        scripts, "generating tutorial...", length=len(scripts)
    )
    for script in iterator:
        _create_tutorial_section(script, src_dir, target_dir)
    
    # Create index
    # String for enumeration of tutorial pages
    include_string = "\n\n".join(
        [f".. include:: {os.path.splitext(script)[0]}.rst"
         for script in scripts]
    )

    index_content = \
f"""
========
Tutorial
========

.. contents::
   :depth: 3

{include_string}
"""
    with open(os.path.join(target_dir, f"index.rst"), "w") as f:
        f.write(index_content)


def _create_tutorial_section(fname, src_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    src_file = os.path.normpath(os.path.join(src_dir, fname))
    # Check if the same tutorial script has been already run
    md5_file = os.path.join(target_dir, f"{fname}.md5")
    if _md5sum_is_current(src_file, md5_file):
        return

    file_conf, script_blocks = parser.split_code_and_text_blocks(src_file)

    # Remove *.py suffix
    base_image_name = os.path.splitext(fname)[0]
    # Locate file in tutorial target directory
    abs_base_image_name = os.path.join(
        os.getcwd(), "tutorial", "target", base_image_name
    )
    image_path_template = abs_base_image_name + "_{0:02}.png"

    fake_main = module_from_spec(spec_from_loader('__main__', None))
    script_vars = {
        "execute_script": True,
        "image_path_iterator": scrapers.ImagePathIterator(image_path_template),
        "src_file": src_file,
        "memory_delta": [],
        "fake_main": fake_main
    }
    tutorial_globals = fake_main.__dict__
    tutorial_globals.update({
        "__doc__": "",
    })
    gallery_conf = copy.deepcopy(DEFAULT_GALLERY_CONF)
    gallery_conf.update({
        "abort_on_example_error": True,
        "src_dir": os.getcwd(),
        "execute_script": True,
        "inspect_global_variables": False,
        "call_memory": (lambda func: (0., func())),
        "image_scrapers": (scrapers.matplotlib_scraper,),
    })
    compiler = codeop.Compile()

    content_rst = ""
    for block_label, block_content, line_no in script_blocks:
        if block_label == "code":
            # Run code and save output images
            code_output = genrst.execute_code_block(
                compiler=compiler,
                block=(block_label, block_content, line_no),
                example_globals=tutorial_globals,
                script_vars=script_vars,
                gallery_conf=gallery_conf,
                file_conf=file_conf
            )
            content_rst += genrst.codestr2rst(
                block_content, lineno=None
            ) + "\n"
            content_rst += code_output

        else:
            content_rst += block_content + "\n\n"
    
    with open(os.path.join(target_dir, f"{base_image_name}.rst"), "w") as file:
        file.write(content_rst)
    
    # Write checksum of file to avoid unnecessary rerun
    with open(md5_file, "w") as file:
        file.write(genrst.get_md5sum(src_file))
    

def _md5sum_is_current(src_file, md5_file):
    if not os.path.exists(md5_file):
        return False
    src_md5 = genrst.get_md5sum(src_file)
    with open(md5_file, "r") as file:
        ref_md5 = file.read()
    return src_md5 == ref_md5
