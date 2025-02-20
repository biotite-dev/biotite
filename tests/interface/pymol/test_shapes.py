import biotite.interface.pymol as pymol_interface


def test_draw_arrow_uniform():
    """
    Draw arrows using the same radius for all arrows.
    The radius is taken is example property here,
    to test if uniform values are properly arrayfied.
    """
    pymol_interface.draw_arrows(
        [(0, 0, 0), (1, 0, 1)], [(1, 0, 0), (1, 0, 1)], radius=0.1
    )


def test_draw_arrow_individual():
    """
    Draw arrows using the same radius for all arrows.
    The radius is taken is example property here,
    to test if uniform values are properly arrayfied.
    """
    pymol_interface.draw_arrows(
        [(0, 0, 0), (1, 0, 1)], [(1, 0, 0), (1, 0, 1)], radius=[0.1, 0.2]
    )
