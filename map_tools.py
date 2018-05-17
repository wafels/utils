from copy import deepcopy

from numpy.random import poisson

import sunpy.map


def _default_apply_function(data):
    return data


def _both_maps_same_dimension(m1, m2):
    return m1.dimensions == m2.dimensions


def _data_add(data1, data2):
    return data1 + data2


def _data_subtract(data1, data2):
    return data1 - data2


def _data_multiply(data1, data2):
    return data1 * data2


def _data_divide(data1, data2):
    return data1 / data2


def map_function_two_arguments(m1, m2, func=_data_add, use_meta=1):
    if not _both_maps_same_dimension(m1, m2):
        raise ValueError("Both input maps must have the same dimensions.")

    if use_meta == 1:
        new_meta = deepcopy(m1.header)
    elif use_meta == 2:
        new_meta = deepcopy(m2.header)
    else:
        raise ValueError("The keyword 'use_meta' must have value that is either 1 or 2.")

    return sunpy.map.Map(func(deepcopy(m1.data), deepcopy(m2.data)), new_meta)


#
# Convenience functions
#
def map_add(m1, m2, use_meta=True):
    return map_function_two_arguments(m1, m2, func=_data_add, use_meta=use_meta)


def map_subtract(m1, m2, use_meta=True):
    return map_function_two_arguments(m1, m2, func=_data_subtract, use_meta=use_meta)


def map_simple_replace(smap, condition, replacement_value):
    """
    Replace pixel values that satisfy condition with the replacement value.

    :param smap:
    :param condition:
    :param replacement_value:
    :return:
    """
    newdata = deepcopy(smap.data)
    newdata[condition] = replacement_value
    return sunpy.map.Map(newdata, smap.meta)


def map_noisy_realization(smap):
    """
    Return a Poisson-noisy version of the input map.

    :param smap:
    :return:
    """
    return sunpy.map.Map(poisson(lam=smap.data), smap.meta)