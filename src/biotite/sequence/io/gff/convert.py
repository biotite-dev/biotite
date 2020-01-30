# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.io.gff"
__author__ = "Patrick Kunzmann"
__all__ = ["get_annotation", "set_annotation"]

from ...annotation import Location, Feature, Annotation


def get_annotation(gff_file):
    """
    Parse a GFF3 file into an :class:`Annotation`.

    The *type* column is used as the :attr:`Feature.key` attribute,
    the locations (``loc``) are taken from the *start*, *end* and
    *strand* columns and  the *attributes* column is parsed into the
    :attr:`Feature.qual` attribute.
    Multiple entries with the same ``ID`` attribute are interpreted
    as the same feature.
    Thus, for entries with the same ``ID``, the *type* and *attributes*
    are only parsed once and the locations are aggregated from each
    entry.
    
    Parameters
    ----------
    gff_file : GFFFile
        The file tro extract the :class:`Annotation` object from.
    
    Returns
    -------
    annotation : Annotation
        The extracted annotation.
    """
    annot = Annotation()
    current_key = None
    current_locs = None
    current_qual = None
    current_id = None
    for _, _, type, start, end, _, strand, _, attrib in gff_file:
        id = attrib.get("ID")
        if id != current_id or id is None:
            # current_key is None, when there is no previous feature
            # (beginning of the file)
            if current_key is not None:
                # Beginning of new feature -> Save previous feature
                annot.add_feature(
                    Feature(current_key, current_locs, current_qual)
                )
            # Track new feature
            current_key = type
            current_locs = [Location(start, end, strand)]
            current_qual = attrib
        else:
            current_locs.append(Location(start, end, strand))
        current_id = id
    # Save last feature
    if current_key is not None:
        annot.add_feature(Feature(current_key, current_locs, current_qual))
    return annot


def set_annotation(gff_file, annotation,
                   seqid=None, source=None, is_stranded=True):
    """
    Write an :class:`Annotation` object into a GFF3 file.

    Each feature will get one entry for each location it has.
    :class:`Feature` objects with multiple locations require the ``ID``
    qualifier in its :attr:`Feature.qual` attribute.
    
    Parameters
    ----------
    gff_file : GFFFile
        The GFF3 file to write into.
    annotation : Annotation
        The annoation which is written to the GFF3 file.
    seqid : str, optional
        The content for the *seqid* column.
    source : str, optional
        The content for the *source* column.
    is_stranded : bool, optional
        If true, the strand of each feature is taken into account.
        Otherwise the *strand* column is filled with '``.``'.
    """
    for feature in sorted(annotation):
        if len(feature.locs) > 1 and "ID" not in feature.qual:
            raise ValueError(
                "The 'Id' qualifier is required "
                "for features with multiple locations"
            )
        ## seqid ##
        if seqid is not None and " " in seqid:
            raise ValueError("The 'seqid' must not contain whitespaces")
        ## source ##
        #Nothing to be done
        ## type ##
        type = feature.key
        ## strand ##
        # Expect same strandedness for all locations
        strand = list(feature.locs)[0].strand if is_stranded else None
        ## score ##
        score = None
        ## attributes ##
        attributes = feature.qual
        # The previous properties are shared by all entries
        # for this feature
        # The following loop handles properties that change with each
        # location
        reverse_order = True if strand == Location.Strand.REVERSE else False
        next_phase = 0
        for loc in sorted(
            feature.locs, key=lambda loc: loc.first, reverse=reverse_order
        ):
            ## start ##
            start = loc.first
            ## end ##
            end = loc.last
            ## strand ##
            strand = loc.strand if is_stranded else None
            ## phase ##
            if type == "CDS":
                phase = next_phase
                # Subtract the length of the location
                next_phase -= loc.last - loc.first + 1
                next_phase %= 3
            else:
                phase = None
            gff_file.append(
                seqid, source, type, start, end,
                score, strand, phase, attributes
            )