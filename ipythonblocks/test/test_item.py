import pytest

from .. import ipythonblocks


def test_basic_api():
    """
    Check that attribute assignment and access works.
    """
    x = ipythonblocks._Item()

    assert x.index_string() == None
    assert x.index is None
    with pytest.raises(AttributeError):
        x.index = (5, 5)

    assert x.row is None
    with pytest.raises(AttributeError):
        x.row = 5

    assert x.col is None
    with pytest.raises(AttributeError):
        x.col = 5

    x = ipythonblocks._Item(index=(5, 6))
    assert x.index == (5, 6)
    assert x.index_string() == '5, 6'
    assert x.row == 5
    assert x.col == 6

    x = ipythonblocks._Item(index=(5, 6), row_major=False)
    assert x.index == (5, 6)
    assert x.index_string() == '5, 6'
    assert x.row == 6
    assert x.col == 5

    x = ipythonblocks._Item(index=(5,))
    assert x.index == (5,)
    assert x.index_string() == '5'
    assert x.row == None
    assert x.col == 5

    x = ipythonblocks._Item(index=(5,), row_major=False)
    assert x.index == (5,)
    assert x.index_string() == '5'
    assert x.row == 5
    assert x.col == None
