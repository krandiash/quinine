import glob
import os

import cytoolz.curried as tz
from cerberus import schema_registry, Validator
from quinine.common.utils import *
from quinine.common.gin import nested_scope_datagroup_gin_dict

tstring = {'type': 'string'}
tinteger = {'type': 'integer'}
tfloat = {'type': 'float'}
tboolean = {'type': 'boolean'}
tlist = {'type': 'list'}
tlistorstring = {'type': ['list', 'string']}
tdict = {'type': 'dict'}

required = {'required': True}
nullable = {'nullable': True}
default = lambda d: {'default': d}
excludes = lambda e: {'excludes': e}
schema = lambda s: {'schema': s}
allowed = lambda a: {'allowed': a}

stlist = lambda s: merge(tlist, schema(merge(tdict, schema(s))))
stlistorstring = lambda s: merge(tlistorstring, schema(merge(tdict, schema(s))))
stdict = lambda s: merge(tdict, schema(s))


def create_and_register_schemas():
    # A schema for gin data
    gin = {'gin': tdict}

    # Loader schema for creating e.g. DataLoaders in torch
    loader = {'loader': stdict({
        'strategy': tstring,
        'batch_size': tinteger,
        'num_workers': tinteger,
        'seed': merge(tinteger, default(0)),
    })}

    # A standard block for data loaders that is used to create a hierarchical schema for dataflows
    datablock = {'source': merge(tstring, allowed(['torchvision', 'imagefolder', 'tfds', 'tfrecord'])),
                 'name': tstring,
                 'shuffle_and_split': tboolean,
                 'load_path': tstring,
                 'seed': tinteger,
                 'heads': merge(tlist, nullable, default(None)),
                 }
    datablock = merge(datablock, loader, gin)

    # Schema for a group: a group is like a slice of data from a particular dataset
    datagroup = merge(datablock,
                      {'split_cmd': merge(tstring, required),
                       'alias': merge(tstring, required),
                       })
    datagroups = {'groups': stlist(datagroup)}

    # A datasets schema for building a list of datasets, with each dataset containing multiple groups
    datasets = {'datasets': stlist(merge(datablock, datagroups))}

    # A dataflow: complete description of all the datasets and groups, their loaders as well as where they are used
    dataflow = merge(datablock,
                     {'train': stdict(merge(datablock,
                                            datasets,
                                            {'loader_level': merge(tstring,
                                                                   required,
                                                                   allowed(['all', 'dataset', 'group']))})),
                      'val': stdict(merge(datablock, datasets,
                                          {'loader_level': merge(tstring,
                                                                 default('group'),
                                                                 allowed(['all', 'dataset', 'group']))})),
                      'test': stdict(merge(datablock, datasets,
                                           {'loader_level': merge(tstring,
                                                                  default('group'),
                                                                  allowed(['all', 'dataset', 'group']))}))
                      })

    # Schema for the duration of training
    duration = {'epochs': merge(tinteger, required, excludes('steps')),
                'steps': merge(tinteger, required, excludes('epochs'))}

    # Schema for the optimizer
    optimizer = {'optimizer': stdict(merge({'name': merge(tstring, required)}, gin))}

    # Schema for the learning rate scheduler
    lr = {'lr': stdict(merge({'scheduler': merge(tstring, required)}, gin))}

    # Trainer: complete description of how to update the model
    trainer = merge(duration, optimizer, lr, gin)

    # Schema for constructing the model
    model = {'source': merge(tstring, required, allowed(['torchvision', 'torch', 'cm', 'keras'])),
             'architecture': merge(tstring, required),
             'finetuning': merge(tboolean, required),
             'pretrained': merge(tboolean, required),
             'heads': merge(tlist, required),
             }
    model = merge(model, gin)

    # Schema for the checkpointer
    checkpointer = {'ckpt_path': merge(tstring, nullable, default(None)),
                    'save_freq': merge(tinteger, required)}
    checkpointer = merge(checkpointer, gin)

    # Schema for resuming model training from Weights and Biases
    wandb_resumer = {'resume': merge(tboolean, default(False)),
                     'run_id': merge(tstring, default(None), nullable),
                     'project': merge(tstring, default(None), nullable),
                     'entity': merge(tstring, default(None), nullable),
                     }
    wandb_resumer = merge(wandb_resumer, gin)

    # Schema for Weights and Biases
    wandb = {'entity': merge(tstring, required),
             'project': merge(tstring, required),
             'group': merge(tstring, default('default')),
             'job_type': merge(tstring, default('training')),
             'ckpt_dir': merge(tstring, default('checkpoints')),
             'dryrun': merge(tboolean, default(False)),
             }
    wandb = merge(wandb, gin)

    # Schema for general settings
    general = {'seed': merge(tinteger, default(0)),
               'module': merge(tstring, required),
               }

    # Schema for Kubernetes
    kubernetes = {}

    # Schema for templating
    templating = {'parent': merge(tstring, nullable, default(None))}

    # Schema for inheritance
    inherit = stlistorstring(merge(tstring, nullable, default(None)))

    # Collect all the schemas that are reusable
    schemas = {
        'general': general,

        'duration': duration,
        'optimizer': optimizer,
        'lr': lr,
        'trainer': trainer,

        'checkpointer': checkpointer,
        'wandb_resumer': wandb_resumer,

        'model': model,

        'kubernetes': kubernetes,
        'templating': templating,
        'inherit': inherit,

        'loader': loader,
        'datablock': datablock,
        'datagroup': datagroup,
        'datagroups': datagroups,
        'datasets': datasets,
        'dataflow': dataflow,

        'gin': gin,

        'wandb': wandb,
    }

    # Register the schemas
    register_schemas(*list(zip(*schemas.items())), verbose=False)

    return schemas


def register_schemas(schema_names, schemas, verbose=True):
    """
    Register a list of schemas, with corresponding names.
    """
    # Register the schemas
    list(map(lambda n, s: schema_registry.add(n, s), schema_names, schemas))

    if verbose:
        # Print
        print("Registered schemas in Cerberus: ")
        list(map(lambda n: print(f'- {n}'), schema_names))


def register_yaml_schemas(path):
    """
    Register all schemas located in a directory.
    Schemas are assumed to be defined in yaml files at path.
    """
    # Get the schema files
    schema_files = glob.glob(os.path.join(path, '*'))
    schema_names = list(map(lambda f: os.path.basename(f).replace(".yaml", ""), schema_files))
    schemas = list(map(compose(autocurry(yaml.load)(Loader=yaml.FullLoader), open), schema_files))

    # Register them
    register_schemas(schema_names, schemas)


def normalize_config(config, schema=None, base_path=''):
    """
    Execute a series of functions on the config that modify it.
    """
    if schema:
        config = Validator(schema).normalized(config)
    config = resolve_templating(config)
    config = resolve_inheritance(config, base_path=base_path)
    if 'dataflow' in config:
        dataflow_config = propagate_parameters_to_datagroups(config.dataflow)
        config = set_in(config, ['dataflow'], dataflow_config)
    config = nested_scope_datagroup_gin_dict(config)
    return config


def resolve_inheritance(config, base_path=''):
    """
    Takes in a config and resolves any inheritance.
    If inheriting, the config will have information about one or more parent configs that should be overwritten
    (those configs may in turn inherit from others).
    This inheritance chain is resolved by recursively merging all the relevant configs.
    """
    if 'inherit' not in config or ('inherit' in config and config['inherit'] is None):
        return config

    inherit_paths = [config['inherit']] if isinstance(config['inherit'], str) else config['inherit']
    inherit_paths = [os.path.abspath(os.path.join(base_path, inherit_path)) for inherit_path in inherit_paths]
    config['inherit'] = inherit_paths

    inherit_configs = [
        # Recurse to resolve inheritance for each inherited config
        resolve_inheritance(
            yaml.load(open(inherit_path),
                      Loader=yaml.FullLoader),
            base_path=os.path.dirname(inherit_path),
        )
        for inherit_path in inherit_paths
    ]

    config = rmerge(*inherit_configs, config)
    return config


def resolve_templating(config, base_path=''):
    """
    Takes in a config and resolves any templating.
    If templating, the config will have information about a parent config that it is overwriting
    (which in turn may itself be templating).
    This templating chain is resolved by recursively merging all the relevant configs.
    """
    if 'templating' not in config or ('templating' in config and config['templating']['parent_yaml'] is None):
        return config

    append_parent = lambda l: [yaml.load(open(os.path.join(base_path, l[0]['templating']['parent_yaml'])),
                                         Loader=yaml.FullLoader)] + l
    construct_hierarchy = lambda l: ignore(errors=Exception,
                                           default=append_parent(l)
                                           )(construct_hierarchy)(append_parent(l))
    config_hierarchy = construct_hierarchy([config])
    config = rmerge(*config_hierarchy)
    return config


def validate_config(config, schema):
    """
    Check if a config file adheres to a schema.
    """
    validator = Validator(schema)
    valid = validator.validate(config)
    if valid:
        return True
    else:
        print("CerberusError: config could not be validated against schema. The errors are,")
        print(validator.errors)
        exit()


def expand_schema_for_gin_configuration(schema):
    """
    Allows the schema to support gin configurability.
    The schema supports gin keys at any level of nesting.
    """
    # Insert a 'gin' key into the dictionaries that don't contain a 'type' key
    predicate = lambda p: 'type' not in p

    # Merge a gin schema into the schema passed in, at every level of nesting
    return nested_dict_walker(iffy(predicate,
                                   lambda v: merge({'gin': tdict}, v)),
                              schema)


def expand_schema_for_inheritance(schema):
    """
    Allows the schema to support inheritance.
    The schema supports configs that (optionally) point to zero or more parent YAML configs
    (using a path) that will be taken as base configurations to be overwritten.
    """
    return merge({'inherit': stlistorstring(merge(tstring,
                                                  nullable,
                                                  default(None))
                                            )
                  }, schema)


def expand_schema_for_templating(schema):
    """
    Allows the schema to support templating.
    The schema supports configs that (optionally) point to a parent YAML config (using a path) that is being overwritten.
    TODO: Deprecate.
    """
    return merge({'templating': stdict({'parent_yaml': merge(tstring,
                                                             nullable,
                                                             default(None))
                                        })
                  }, schema)


def autoexpand_schema(schema):
    """
    Automatically expands the schema to support
    - gin configuration
    - templating
    """
    schema = expand_schema_for_gin_configuration(schema)
    schema = expand_schema_for_templating(schema)
    schema = expand_schema_for_inheritance(schema)
    return schema


def propagate_parameters_to_datagroups(dataflow_config):
    """
    Given a dataflow config, it is likely that parameters were defined 'globally' e.g. setting the dataset's source
    for all datasets and groups. This function propagates parameters from higher levels down to the lowest, group level.
    The propagation consolidates parameters at the following levels:
    - dataflow
    - train(/val/test)
    - datasets[i]
    - groups[j]
    into the parameters that are applicable to groups[j].
    """

    def construct_group_dict(group_path, config):
        """
        Given a config and a path that points to a data group, compute the data group's updated parameters.
        The group_path is a list of keys and indices e.g. ['train', 'datasets', 1, 'groups', 0]
        that can be followed to reach a group's config.
        """
        # Find (almost) all prefixes of the group path
        all_paths = list(map(compose(list,
                                     tz.take(seq=group_path)),
                             range(1, len(group_path))
                             )
                         )

        # Filter to exclude paths that point to lists
        paths_to_merge = list(filter(lambda p: isinstance(last(p[1]), str),
                                     pairwise(all_paths)
                                     )
                              )
        # Find all the (mid-level) dicts that the filtered paths point to
        mid_level_dicts = list(map(lambda p: tz.keyfilter(lambda k: k != last(p[1]),
                                                          tz.get_in(p[0], config)),
                                   paths_to_merge))

        # Merge parameters at all levels to get a single parameter set for the group
        def dmerge(*args):
            if all(is_mapping, *args):
                return Munch(tz.merge(*args))
            else:
                return tz.last(*args)

        group_dict = tz.merge_with(
            dmerge,
            tz.keyfilter(lambda k: k not in ['train', 'val', 'test'], config),  # top-level dict
            *mid_level_dicts,  # mid-level dicts
            tz.get_in(group_path, config)  # bottom-level dict
        )

        return group_dict

    def get_all_group_paths(config, following=()):
        """
        Given a config, constructs paths to all the leaf nodes, truncating them one level below the 'groups' key.
        """
        if isinstance(config, dict) or isinstance(config, Munch):
            if 'groups' in list(butlast(following)):
                return [[]]
            return list(cat(map(lambda t: list(map(lambda p: [t[0]] + p,
                                                   get_all_group_paths(t[1], list(following) + [t[0]])
                                                   )),
                                iteritems(config)))
                        )

        elif isinstance(config, list):
            return list(cat(map(lambda t: list(map(lambda p: [t[0]] + p,
                                                   get_all_group_paths(t[1], list(following) + [t[0]])
                                                   )),
                                enumerate(config)))
                        )
        else:
            return [[]]

    # Find all the group paths
    group_paths = list(filter(lambda p: 'groups' in p,
                              get_all_group_paths(dataflow_config)
                              )
                       )

    # Construct the group dict for each group path
    group_dicts = list(map(lambda p: construct_group_dict(p,
                                                          dataflow_config),
                           group_paths)
                       )

    # Update the dataflow_config with all the group dicts
    updated_dataflow_config = compose(*list(map(autocurry(lambda p, d, c: set_in(c, p, d)),
                                                group_paths,
                                                group_dicts,
                                                )
                                            ))(dataflow_config)

    return updated_dataflow_config


if __name__ == '__main__':
    register_yaml_schemas(path='configs/schemas/base/')
    print()
