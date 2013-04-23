import os
import uuid
import pytest

from .. import ipythonblocks


def fake_uuid():
    return 'abc'


@pytest.fixture
def basic_array():
    return ipythonblocks._NDArray(shape=(6, 5))


def test_basic_api(basic_array):
    """
    Check that inputs are going to the right attributes and that assignment
    works when it should and not when it shouldn't.

    """
    ba = basic_array

    assert ba.width == 5
    with pytest.raises(AttributeError):
        ba.width = 20

    assert ba.height == 6
    with pytest.raises(AttributeError):
        ba.height = 20

    assert ba.shape == (5, 6)


def test_view(basic_array):
    """
    Check that slicing _NDArrays returns a view and not a copy.

    """
    ba = basic_array
    na = ba[:2, :2]

    na[1, 1].dummy = 5

    for item in (na[1, 1], ba[1, 1]):
        assert hasattr(item, 'dummy')
        assert item.dummy == 5


def test_iter():
    """
    Test that we do complete, row-first iteration.

    """
    ba = ipythonblocks._NDArray(shape=(2, 2))

    coords = ((0, 0), (0, 1), (1, 0), (1, 1))

    for item, c in zip(ba, coords):
        assert item.row == c[0]
        assert item.col == c[1]


def test_view_coords(basic_array):
    """
    Check that that views have appropriate coordinates.

    """
    na = basic_array[-2:, -2:]

    coords = ((0, 0), (0, 1), (1, 0), (1, 1))

    for item, c in zip(na, coords):
        assert item.row == c[0]
        assert item.col == c[1]


def test_copy(basic_array):
    """
    Check that _NDArray.copy returns a totally independent copy (not a view).

    """
    ba = basic_array
    na = ba[:2, :2].copy()

    na[1, 1].dummy = 5

    assert na[1, 1].dummy == 5
    assert not hasattr(ba[1, 1], 'dummy')


def test_str(basic_array):
    """
    Test the _NDArray.__str__ method used with print.

    """
    ba = basic_array

    s = os.linesep.join(['_NDArray', 'Shape: (6, 5)'])

    assert ba.__str__() == s


def test_bad_index(basic_array):
    """
    Test for the correct errors with bad indices.

    """
    ba = basic_array

    with pytest.raises(IndexError):
        ba[1, 2, 3, 4]

    with pytest.raises(IndexError):
        ba[{4: 5}]

    with pytest.raises(TypeError):
        ba[1, ]


def test_getitem(basic_array):
    """
    Exercise a bunch of different indexing.

    """
    ba = basic_array

    # single item
    item = ba[1, 2]

    assert isinstance(item, ipythonblocks._Item)
    assert item.row == 1
    assert item.col == 2

    # single row
    na = ba[2]

    assert isinstance(na, ipythonblocks._NDArray)
    assert na.shape == (1, ba.width)

    # two rows
    na = ba[1:3]

    assert isinstance(na, ipythonblocks._NDArray)
    assert na.shape == (2, ba.width)

    # one row via a slice
    na = ba[2, :]

    assert isinstance(na, ipythonblocks._NDArray)
    assert na.shape == (1, ba.width)

    # one column
    na = ba[:, 2]

    assert isinstance(na, ipythonblocks._NDArray)
    assert na.shape == (ba.height, 1)

    # 2 x 2 subarray
    na = ba[:2, :2]

    assert isinstance(na, ipythonblocks._NDArray)
    assert na.shape == (2, 2)

    # strided slicina
    na = ba[::3, ::3]

    assert isinstance(na, ipythonblocks._NDArray)
    assert na.shape == (2, 2)

    # one column / one row with a -1 index
    # testina fix for #7
    na = ba[-1, :]

    assert isinstance(na, ipythonblocks._NDArray)
    assert na.shape == (1, ba.width)

    na = ba[1:4, -1]

    assert isinstance(na, ipythonblocks._NDArray)
    assert na.shape == (3, 1)


def test_setitem(basic_array):
    """
    Test assigning items in arrays.

    """
    ba = basic_array
    value = 'new'

    # single block
    ba[0, 0] = value
    assert ba[0, 0] == value

    # single row
    ba = ipythonblocks._NDArray(shape=ba.shape)
    ba[2] = value
    for item in ba[2]:
        assert item == value

    # two rows
    ba = ipythonblocks._NDArray(shape=ba.shape)
    ba[3:5] = value
    for item in ba[3:5]:
        assert item == value

    # one row via a slice
    ba = ipythonblocks._NDArray(shape=ba.shape)
    ba[1, :] = value
    for item in ba[1, :]:
        assert item == value

    # one column
    ba = ipythonblocks._NDArray(shape=ba.shape)
    ba[:, 5] = value
    for item in ba[:, 5]:
        assert item == value

    # 2 x 2 subarray
    ba = ipythonblocks._NDArray(shape=ba.shape)
    ba[:2, :2] = value
    for item in ba[:2, :2]:
        assert item == value

    # strided slicing
    ba = ipythonblocks._NDArray(shape=ba.shape)
    ba[::3, ::3] = value
    for item in ba[::3, ::3]:
        assert item == value


def test_to_text(capsys):
    """
    Test using the _NDArray.to_text method.

    """
    ba = ipythonblocks._NDArray(shape=(1, 2))

    ba[0, 0].rgb = (1, 2, 3)
    ba[0, 1].rgb = (4, 5, 6)

    ref = ['# shape (..., height, width)',
           '1 2',
           '# row-major',
           'True',
           '# index data',
           '0 0',
           '0 1']
    ref = os.linesep.join(ref) + os.linesep

    ba.to_text()
    out, err = capsys.readouterr()

    assert out == ref
