from ast import literal_eval
from .types import ImageSize3D, ImageSpacing3D

def parse_size_3D(s: str) -> ImageSize3D:
    """
    returns: a ImageSize3D object.
    args:
        s: the string to parse.
    """
    # Evaluate string literal.
    try:
        s = literal_eval(s)
    except SyntaxError:
        raise TypeError(f"Couldn't parse '{s}' into type '{ImageSize3D}', got syntax error.")

    # Check type.
    if not isinstance(s, tuple):
        raise TypeError(f"Couldn't parse '{s}' into type '{ImageSize3D}', got type '{type(s)}', expected tuple.")

    # Check length.
    if not len(s) == 3:
        raise TypeError(f"Couldn't parse '{s}' into type '{ImageSize3D}', got length '{len(s)}', expected 3.")

    # Check elements.
    for i in s:
        if not isinstance(i, int):
            raise TypeError(f"Couldn't parse '{s}' into type '{ImageSize3D}', got element '{i}' of type '{type(i)}', expected int.")

    return s

def parse_spacing_3D(s: str) -> ImageSpacing3D:
    """
    returns: a ImageSpacing3D object.
    args:
        s: the string to parse.
    """
    # Evaluate string literal.
    try:
        s = literal_eval(s)
    except SyntaxError:
        raise TypeError(f"Couldn't parse '{s}' into type '{ImageSpacing3D}', got syntax error.")

    # Check type.
    if not isinstance(s, tuple):
        raise TypeError(f"Couldn't parse '{s}' into type '{ImageSpacing3D}', got type '{type(s)}', expected tuple.")

    # Check length.
    if not len(s) == 3:
        raise TypeError(f"Couldn't parse '{s}' into type '{ImageSpacing3D}', got length '{len(s)}', expected 3.")

    # Check elements.
    for i in s:
        if not isinstance(i, (int, float)):
            raise TypeError(f"Couldn't parse '{s}' into type '{ImageSpacing3D}', got element '{i}' of type '{type(i)}', expected int or float.")

    return s