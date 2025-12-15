# -*- coding: utf-8 -*-
import re

def natural_sort_key(s):
    """Sort string with natural order.

    Parameters
    ----------
    s: list(str)
        the list of strings.

    Returns
    -------
    list
        return the list with natural order.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


