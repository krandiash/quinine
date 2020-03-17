"""
Some common utilities.
"""

import yaml
from funcy import *
from munch import Munch


def rmerge(*colls):
    """
    Recursively merge an arbitrary number of collections.
    For conflicting values, later collections to the right are given priority.
    Note that this function treats sequences as a normal value and sequences are not merged.

    Uses:
    - merging config files
    """
    if isinstance(colls, tuple) and len(colls) == 1:
        # A squeeze operation since merge_with generates tuple(list_of_objs,)
        colls = colls[0]
    if all(is_mapping, colls):
        # Merges all the collections, recursively applies merging to the combined values
        return merge_with(rmerge, *colls)
    else:
        # If colls does not contain mappings, simply pick the last one
        return last(colls)


def prettyprint(s):
    if hasattr(s, '__dict__'):
        print(yaml.dump(s.__dict__))
    elif isinstance(s, dict):
        print(yaml.dump(s))
    else:
        print(s)


def allequal(seq):
    return len(set(seq)) <= 1


@autocurry
def listmap(fn, seq):
    return list(map(fn, seq))


@autocurry
def prefix(s, p):
    if isinstance(s, str):
        return f'{p}{s}'
    elif isinstance(s, list):
        return list(map(prefix(p=p), s))
    else:
        raise NotImplementedError


@autocurry
def postfix(s, p):
    if isinstance(s, str):
        return f'{s}{p}'
    elif isinstance(s, list):
        return list(map(postfix(p=p), s))
    else:
        raise NotImplementedError


@autocurry
def surround(s, pre, post):
    return postfix(prefix(s, pre), post)


def nested_map(f, *args):
    """ Recursively transpose a nested structure of tuples, lists, and dicts """
    assert len(args) > 0, 'Must have at least one argument.'

    arg = args[0]
    if isinstance(arg, tuple) or isinstance(arg, list):
        return [nested_map(f, *a) for a in zip(*args)]
    elif isinstance(arg, dict):
        return {
            k: nested_map(f, *[a[k] for a in args])
            for k in arg
        }
    else:
        return f(*args)


@autocurry
def walk_values_rec(f, coll):
    """
    Similar to funcy's walk_values, but does so recursively, including mapping f over lists.
    """
    if is_mapping(coll):
        return f(walk_values(walk_values_rec(f), coll))
    elif is_list(coll):
        return f(list(map(walk_values_rec(f), coll)))
    else:
        return f(coll)


@autocurry
def nested_dict_walker(fn, coll):
    """
    Apply a function over the mappings contained in coll.
    """
    return walk_values_rec(iffy(is_mapping, fn), coll)


def get_all_leaf_paths(coll):
    """
    Returns a list of paths to all leaf nodes in a nested dict.
    Paths can travel through lists and the index is inserted into the path.
    """
    if isinstance(coll, dict) or isinstance(coll, Munch):
        return list(cat(map(lambda t: list(map(lambda p: [t[0]] + p,
                                               get_all_leaf_paths(t[1])
                                               )),
                            iteritems(coll)))
                    )

    elif isinstance(coll, list):
        return list(cat(map(lambda t: list(map(lambda p: [t[0]] + p,
                                               get_all_leaf_paths(t[1])
                                               )),
                            enumerate(coll)))
                    )
    else:
        return [[]]
