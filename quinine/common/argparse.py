from argparse import ArgumentParser
from funcy import *
import cytoolz as tz

from quinine.quinfig import Quinfig
from quinine.quinfig import get_all_leaf_paths


class QuinineArgumentParser(ArgumentParser):
    """
    Class for replacing the standard argparse.ArgumentParser.

    In addition to the standard functionality of ArgumentParser,
    - includes an argument for '--config', which passes in a YAML file containing the configuration parameters
    - automatically adds arguments to the ArgumentParser from a provided schema,
    allowing you to override arguments in your YAML directly from the command line
    """
    types = {'string': str,
             'integer': int,
             'float': float,
             'dict': dict,
             'boolean': bool,
             'list': list,
             }

    def __init__(self, schema=None, **kwargs):
        super(QuinineArgumentParser, self).__init__(**kwargs)

        # Add a default argument for the path to the YAML configuration file that will be passed in
        self.add_argument('--config', type=str, required=True, help="YAML configuration file.")
        self.schema = schema

        if self.schema is not None:
            # Populate the argument parser with arguments from the schema
            paths_to_type = list(filter(lambda l: l[-1] == 'type', get_all_leaf_paths(self.schema)))
            type_lookup = dict([(tuple(filter(lambda e: e != 'schema', e[:-1])), tz.get_in(e, schema))
                                for e in paths_to_type])

            valid_params = self.get_all_params(schema)
            for param in valid_params:
                self.add_argument(f'--{".".join(param)}',
                                  type=self.types[type_lookup[param]])

    @staticmethod
    def get_all_params(schema):
        # Find all leaf paths in the schema, then truncate the last key from each path
        # and remove the 'schema' key if it occurs anywhere in the path
        # TODO: expand the list of criteria in the inner lambda
        candidate_parameters = list(set(map(lambda l: tuple(filter(lambda e: e != 'schema' and e != 'allowed', l[:-1])),
                                            get_all_leaf_paths(schema))
                                        )
                                    )

        # Remove prefix paths from the candidate parameters,
        # e.g. when ['general', 'seed'] and ['general'] both occur, remove ['general']
        valid_parameters = set()
        all_subpaths = set()
        for path in sorted(candidate_parameters, key=lambda l: len(l), reverse=True):
            # If the path isn't in the set of subpaths seen so far, it's a valid param (because paths are sorted)
            if path not in all_subpaths:
                valid_parameters.add(path)
                # Add all subpaths in for this
                for i in range(1, len(path)):
                    all_subpaths.add(path[:i])

        return valid_parameters

    def parse_quinfig(self):
        # Parse all the arguments
        args = self.parse_args()
        override_args = dict(select_values(lambda v: v is not None,
                                           omit(args.__dict__, ['config'])
                                           )
                             )

        # Trick: first load the config without a schema
        quinfig = Quinfig(config_path=args.config)

        # Replace all the arguments passed into command line
        if len(override_args) > 0:
            print(f"Overriding arguments in {args.config} from command line.")
        for param, val in override_args.items():
            param_path = param.split(".")
            old_val = tz.get_in(param_path, quinfig)
            print(f"> ({param}): {old_val} --> {val}")
            quinfig = tz.assoc_in(quinfig, param_path, val)

        # Load the config again, this time with the schema
        return Quinfig(config=quinfig.__dict__ if isinstance(quinfig, Quinfig) else quinfig, schema=self.schema)
