import os
import uuid
import pytest

from .. import ipythonblocks


def fake_uuid():
    return 'abc'


@pytest.fixture
def basic_grid():
    return ipythonblocks.BlockGrid(5, 6, (1, 2, 3), 20, True)


def test_basic_api(basic_grid):
    """
    Check that inputs are going to the right attributes and that assignment
    works when it should and not when it shouldn't.

    """
    bg = basic_grid

    assert bg.width == 5
    with pytest.raises(AttributeError):
        bg.width = 20

    assert bg.height == 6
    with pytest.raises(AttributeError):
        bg.height = 20

    assert bg.shape == (5, 6)
    assert bg.block_size == 20
    assert bg.lines_on is True


def test_grid_init(basic_grid):
    """
    Test that the grid is properly initialized.

    """
    bg = basic_grid

    for r in range(bg.height):
        for c in range(bg.width):
            assert bg[r, c].size == 20
            assert bg[r, c].red == 1
            assert bg[r, c].green == 2
            assert bg[r, c].blue == 3
            assert bg[r, c].row == r
            assert bg[r, c].col == c


def test_change_block_size(basic_grid):
    """
    Test that all blocks are properly resized when changing the
    BlockGrid.block_size attribute.

    """
    bg = basic_grid

    bg.block_size = 10
    assert bg.block_size == 10

    for block in bg:
        assert block.size == 10


def test_change_lines_on(basic_grid):
    """
    Test changing the BlockGrid.lines_on attribute.

    """
    bg = basic_grid

    assert bg.lines_on is True

    bg.lines_on = False
    assert bg.lines_on is False

    with pytest.raises(ValueError):
        bg.lines_on = 5

    with pytest.raises(ValueError):
        bg.lines_on = 'asdf'


def test_str(basic_grid):
    """
    Test the BlockGrid.__str__ method used with print.

    """
    bg = basic_grid

    s = os.linesep.join(['BlockGrid', 'Shape: (5, 6)'])

    assert bg.__str__() == s


def test_repr_html(monkeypatch):
    """
    HTML repr should be the same for a 1, 1 BlockGrid as for a single Block.
    (As long as the BlockGrid border is off.)

    """
    bg = ipythonblocks.BlockGrid(1, 1, lines_on=False)

    monkeypatch.setattr(uuid, 'uuid4', fake_uuid)

    assert bg._repr_html_() == bg[0, 0]._repr_html_()


def test_bad_colors(basic_grid):
    """
    Make sure this gets the right error when trying to assign something
    other than three integers.

    """
    with pytest.raises(ValueError):
        basic_grid[0, 0] = (1, 2, 3, 4)


def test_to_text(capsys):
    """
    Test using the BlockGrid.to_text method.

    """
    bg = ipythonblocks.BlockGrid(2, 1, block_size=20)

    bg[0, 0].rgb = (1, 2, 3)
    bg[0, 1].rgb = (4, 5, 6)

    ref = ['# width height',
           '2 1',
           '# block size',
           '20',
           '# initial color',
           '0 0 0',
           '# row column red green blue',
           '0 0 1 2 3',
           '0 1 4 5 6']
    ref = os.linesep.join(ref) + os.linesep

    bg.to_text()
    out, err = capsys.readouterr()

    assert out == ref
