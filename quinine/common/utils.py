"""
Some common utilities.
"""

import yaml
from funcy import *
from munch import Munch
import cytoolz as tz


def difference(*colls):
    """
    Find the keys that have different values in an arbitrary number of (nested) collections. Any key
    that differs in at least 2 collections is considered to fit this criterion.
    """

    # Get all the leaf paths for each collection: make each path a tuple
    leaf_paths_by_coll = list(map(lambda c: list(map(tuple, get_all_leaf_paths(c))), colls))

    # Find the union of all leaf paths: merge all the paths and keep only the unique paths
    union_leaf_paths = list(distinct(concat(*leaf_paths_by_coll)))

    # Get the values corresponding to these leaf paths in every collection: if a leaf path doesn't exist, assumes None
    values_by_coll = list(map(lambda lp: list(map(lambda coll: tz.get_in(lp, coll), colls)), union_leaf_paths))

    # Filter out the leaf paths that have identical values across the collections
    keep_leaf_paths = list(map(0, filter(lambda t: not allequal(t[1]), zip(union_leaf_paths, values_by_coll))))
    keep_values = list(map(1, filter(lambda t: not allequal(t[1]), zip(union_leaf_paths, values_by_coll))))

    # Rearrange to construct a list of dictionaries -- one per original collection.
    # Each of these dictionaries maps a 'kept' leaf path to its corresponding
    # value in the collection
    differences = list(map(lambda vals: dict(zip(keep_leaf_paths, vals)), list(zip(*keep_values))))

    return differences



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


def get_all_paths(coll, prefix_path=(), stop_at=None, stop_below=None):
    """
    Given a collection, by default returns paths to all the leaf nodes.
    Use stop_at to truncate paths at the given key.
    Use stop_below to truncate paths one level below the given key.
    """
    assert stop_at is None or stop_below is None, 'Only one of stop_at or stop_below can be used.'
    if stop_below is not None and stop_below in str(last(butlast(prefix_path))):
        return [[]]
    if stop_at is not None and stop_at in str(last(prefix_path)):
        return [[]]
    if isinstance(coll, dict) or isinstance(coll, Munch) or isinstance(coll, list):
        if isinstance(coll, dict) or isinstance(coll, Munch):
            items = iteritems(coll)
        else:
            items = enumerate(coll)

        return list(cat(map(lambda t: list(map(lambda p: [t[0]] + p,
                                               get_all_paths(t[1],
                                                             prefix_path=list(prefix_path) + [t[0]],
                                                             stop_at=stop_at,
                                                             stop_below=stop_below)
                                               )),
                            items))
                    )
    else:
        return [[]]


def get_only_paths(coll, pred, prefix_path=(), stop_at=None, stop_below=None):
    """
    Get all paths that satisfy the predicate fn pred.
    First gets all paths and then filters them based on pred.
    """
    all_paths = get_all_paths(coll, prefix_path=prefix_path, stop_at=stop_at, stop_below=stop_below)
    return list(filter(pred, all_paths))

if __name__ == '__main__':
    coll1 = {'a': 1,
             'b': 2,
             'c': {'d': 12}}
    coll2 = {'a': 1,
             'b': 2,
             'c': {'d': 13}}
    coll3 = {'a': 1,
             'b': 3,
             'c': {'d': 14},
             'e': 4}

    difference(coll1, coll2, coll3)