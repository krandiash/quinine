import gin
from funcy import *
from quinine.common.utils import nested_dict_walker, prefix


def register_module_with_gin(module, module_name=None):
    """
    Register all the callables in a single module with gin.

    A useful way to add gin configurability to a codebase without explicilty using the @gin.configurable decorator.
    """
    module_name = module.__name__ if module_name is None else module_name

    for attr in dir(module):
        if callable(getattr(module, attr)):
            setattr(module, attr, gin.configurable(getattr(module, attr), module=module_name))


def scope_datagroup_gin_dict(coll):
    """
    Rename the augmentations gin dict with the alias of the dataset.
    """

    if 'alias' in coll and 'gin' in coll and is_mapping(coll['gin']):
        coll['gin'] = walk_keys(prefix(p=f"{coll['alias'].replace('.', '_')}/"), coll['gin'])
    return coll


def nested_scope_datagroup_gin_dict(coll):
    """
    Apply the renamer over a nested dict, e.g. derived from a yaml.
    """
    return nested_dict_walker(scope_datagroup_gin_dict, coll)


def gin_dict_parser(coll):
    """
    Use for parsing collections that may contain a 'gin' key.
    The 'gin' key is assumed to map to either a dict or str value that contains gin bindings.
    e.g.
    {'gin': {'Classifier.n_layers': 2, 'Classifier.width': 3}}
    or
    {'gin': 'Classifier.n_layers = 2\nClassifier.width = 3'}
    """
    if 'gin' in coll:
        if is_mapping(coll['gin']):
            gin.parse_config("".join(map(lambda t: f'{t[0]} = {t[1]}\n', iteritems(coll['gin']))))
        elif isinstance(coll['gin'], str):
            gin.parse_config(coll['gin'])
    return coll


def nested_gin_dict_parser(coll):
    """
    Use for parsing nested collections that may contain a 'gin' key.
    The 'gin' key is assumed to map to a dict value that contains gin bindings (see gin_dict_parser).

    Enables support for gin keys in yaml files.
    """
    return nested_dict_walker(gin_dict_parser, coll)
