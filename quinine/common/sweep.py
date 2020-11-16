import itertools
import re
import os
import typing as typ
from collections import namedtuple
from copy import deepcopy

import cytoolz as tz
import yaml
from funcy import *
from toposort import toposort

from quinine.common.utils import get_only_paths, allequal
from quinine.quinfig import Quinfig

# Parameter = namedtuple('Parameter', 'path dotpath value')
# SweptParameter = namedtuple('SweptParameter', 'path sweep')
SweptDisjointParameter = namedtuple('SweptDisjointParameter', 'path disjoint')
SweptProductParameter = namedtuple('SweptProductParameter', 'path product')
SweptDefaultParameter = namedtuple('SweptDefaultParameter', 'path default')


class Parameter:

    def __init__(self, path=None, dotpath=None, value=None):
        self.path = path
        self.dotpath = dotpath
        self.value = value

    def __eq__(self, other):
        if hasattr(other, 'dotpath'):
            return self.dotpath == other.dotpath
        else:
            return self.dotpath == other

    def __repr__(self):
        return f'Parameter(path={self.path}, dotpath={self.dotpath}, value={self.value})'


class SweptParameter:

    def __init__(self, path=None, sweep=None):
        self.path = path
        self.sweep = sweep

    def __getitem__(self, item):
        if item == 0:
            return self.path
        elif item == 1:
            return self.sweep

    def __repr__(self):
        return f'SweptParameter(path={self.path}, sweep={self.sweep})'


class QuinSweep:
    """
    The QuinSweep class can be used to
    - process a parameter sweep, with support for complex conditional sweeps
    - generate Quinfigs for each setting of the parameter sweep

    e.g.

    """
    # inside any parameter, specify a sweep with the special ~ character
    SWEEP_PREFIX = '~'

    # 'default' *must* be the last entry in this list
    SWEEP_TOKENS = ['disjoint', 'product'] + ['default']

    @staticmethod
    def get_parameter_path(token_path: typ.List[str]) -> typ.List[str]:
        """
        Given the path to a sweep token
        e.g.
        ['model', 'architecture', 'n_layers', '~product']
        which points to the sweep token '~product' for the 'n_layers' parameter

        get_parameter_path returns the path to the parameter on which the sweep token was applied,
        i.e.
        ['model', 'architecture', 'n_layers'].
        """
        return compose(tuple, butlast)(token_path)

    @staticmethod
    def parse_sweep_config(sweep_config) -> typ.List[typ.List[str]]:
        """
        Takes in a sweep config and determines paths to the sweep tokens in the config.

        Uses the fact that all sweep tokens in the config are prefixed by the SWEEP_PREFIX (~) character.

        Each token path
        """
        # To identify the swept parameters,
        # use the fact that any parameter sweep must use the special prefix (~ by default)
        token_paths = get_only_paths(sweep_config,
                                     pred=lambda p: any(lambda s: QuinSweep.SWEEP_PREFIX in str(s), p),
                                     stop_at=QuinSweep.SWEEP_PREFIX)

        # As output, we get a list of paths that point to locations of all the ~ tokens in the config
        token_paths = list(map(tuple, token_paths))

        # Confirm that all the tokens followed by ~ are correctly specified
        all_tokens_ok = all(lambda s: s.startswith(QuinSweep.SWEEP_PREFIX) and
                                      s.strip(QuinSweep.SWEEP_PREFIX) in QuinSweep.SWEEP_TOKENS,
                            map(last, token_paths)
                            )
        assert all_tokens_ok, f'Unknown token: sweep config failed parsing. ' \
                              f'Only tokens {QuinSweep.SWEEP_TOKENS} are allowed.'

        return token_paths

    @staticmethod
    def parse_fixed_parameters(sweep_config):
        # Extract the locations of the other non-swept, fixed parameters
        # use the fact that the other parameters must not be prefixed by the special prefix
        fixed_parameters = get_only_paths(sweep_config,
                                          pred=lambda p: all(lambda s: QuinSweep.SWEEP_PREFIX not in str(s), p))

        # Make Parameter objects
        fixed_parameters = list(map(lambda tp: Parameter(tp,
                                                         f'{".".join(map(str, tp))}.0',
                                                         get_in(sweep_config, tp)),
                                    fixed_parameters)
                                )
        return fixed_parameters

    @staticmethod
    def path_to_dotpath(path: typ.List) -> str:
        """
        Takes in any path
        e.g.
        ['model', 'architecture', 'n_layers']
        and converts it to a dotpath
        i.e.
        'model.architecture.n_layers'.
        """
        return ".".join(path)

    @staticmethod
    def dotpath_to_path(dotpath):
        """
        Takes in a dotpath
        e.g.
        'model.architecture.n_layers'
        and converts it to a path
        e.g.
        ['model', 'architecture', 'n_layers'].

        Only gin parameters are allowed to have dots in their name, and
        this fn correctly handles conversion for those cases:
        e.g.
        taking 'model.gin.architecture.n_layers' and converting it into ['model', 'gin', 'architecture.n_layers'].
        """
        # Split up the dotpath
        split = dotpath.split(".")

        # Handle the case where the gin parameter has a dot
        if len(split) > 2 and split[-3] == 'gin':
            split = split[:-2] + [".".join(split[-2:])]

        return split

    @staticmethod
    def parse_ref_dotpath(dotpath):
        """
        Reference dotpaths consist of this.is.a.parameter.idx.this.is.another.parameter._.parameter.idx (_ indicates any idx)

        Returns a list of (dotpath, idx) tuples.
        """
        return re.findall('((?:\w[^.]*)+(?:\.\w[^.]*)*?)\.(\d+|_)', dotpath)

    @staticmethod
    def param_comparator(p, q):
        """
        A comparison operator that checks which of two sweeps to apply first.

        The comparison relies on the conditional dependencies expressed in both sweeps.
        """
        # Create dotpaths for both sweeps
        p_dotpath = ".".join(p.path)
        q_dotpath = ".".join(q.path)

        # Get the keys that each sweep refers (these are the conditional dependencies that the sweep has)
        if isinstance(p, SweptDisjointParameter):
            p_ref_dotpaths = p.disjoint.keys()
        elif isinstance(p, SweptProductParameter):
            p_ref_dotpaths = p.product.keys()
        else:
            raise NotImplementedError
        if isinstance(q, SweptDisjointParameter):
            q_ref_dotpaths = q.disjoint.keys()
        elif isinstance(q, SweptProductParameter):
            q_ref_dotpaths = q.product.keys()
        else:
            raise NotImplementedError

        # Each key is a combination of dotpaths and index references: parse them to extract the dotpaths
        p_ref_parsed_dotpaths = list(
            mapcat(compose(list, autocurry(map)(0), QuinSweep.parse_ref_dotpath), p_ref_dotpaths))
        q_ref_parsed_dotpaths = list(
            mapcat(compose(list, autocurry(map)(0), QuinSweep.parse_ref_dotpath), q_ref_dotpaths))

        # Check if any reference dotpath in q matches p or vice-versa
        if any(lambda e: p_dotpath == e or p_dotpath.endswith(f'.{e}'), q_ref_parsed_dotpaths):
            return 1
        elif any(lambda e: q_dotpath == e or q_dotpath.endswith(f'.{e}'), p_ref_parsed_dotpaths):
            return -1
        else:
            return 0

    @staticmethod
    def is_product_subtype(subtype):
        return 'SweptProductParameter' in subtype.__name__

    @staticmethod
    def is_disjoint_subtype(subtype):
        return 'SweptDisjointParameter' in subtype.__name__

    def expand_partial_dotpath(self, partial_dotpath):
        """
        Given a partial dotpath, expand the dotpath to yield a full dotpath from the root to the parameter.
        """
        # This assertion checks if the reference made by the partial dotpath points to another parameter *uniquely*.
        # If not, it's impossible to know which parameter was being referenced.
        assert one(lambda k: k.endswith(partial_dotpath),
                   self.swept_parameters_dict.keys())  # swept_parameter_dict.keys() -> all_dotpaths

        # Complete the dotpath, yielding the full dotpath that points to the location of the parameter
        return list(filter(lambda t: t[1] is True,
                           map(lambda k: (k, k.endswith(partial_dotpath)),
                               self.swept_parameters_dict.keys())
                           )
                    )[0][0]

    def expand_reference(self, reference):
        # Parse the reference dotpath
        parsed_ref_dotpath = self.parse_ref_dotpath(reference)

        # Map to expand the dotpaths in the reference
        parsed_ref_dotpath = list(map(lambda t: (self.expand_partial_dotpath(t[0]), t[1]), parsed_ref_dotpath))

        # Join back
        return ".".join(list(cat(parsed_ref_dotpath)))

    def __init__(self,
                 sweep_config_path=None,
                 schema_path=None,
                 sweep_config=None,
                 schema=None
                 ):

        # Load up the sweep config
        if sweep_config is None:
            assert sweep_config_path is not None, 'Please pass in either sweep_config or sweep_config_path.'
            assert sweep_config_path.endswith('.yaml'), 'Must use a YAML file for the sweep_config.'
            sweep_config = yaml.load(open(sweep_config_path),
                                     Loader=yaml.FullLoader)

        # First, extract the locations of the sweeps being performed
        self.sweep_paths = QuinSweep.parse_sweep_config(sweep_config)

        if len(self.sweep_paths) == 0:
            # No sweep: just return the single Quinfig
            self.quinfigs = [Quinfig(config=sweep_config)]
            print(f"Generated {len(self.quinfigs)} quinfig(s) successfully.")
            return

        # Create list of paths to all the parameters that are being swept
        self.swept_parameter_paths = list(distinct(map(QuinSweep.get_parameter_path,
                                                       self.sweep_paths),
                                                   key=tuple)
                                          )

        # Next, extract the fixed parameters from the sweep config
        self.fixed_parameters = QuinSweep.parse_fixed_parameters(sweep_config)

        # Next, fetch the SweptParameter named tuples after creating them
        self.swept_parameters, \
        self.swept_disjoint_parameters, \
        self.swept_product_parameters, \
        self.swept_default_parameters = \
            self.fetch_swept_parameters(sweep_config, self.sweep_paths, self.swept_parameter_paths)

        # For convenience, create a lookup table for the swept parameters from their dotpaths
        self.swept_parameters_dict = dict(zip(map(lambda sp: ".".join(sp.path),
                                                  self.swept_parameters),
                                              self.swept_parameters)
                                          )

        # Expand all the dotpaths in any conditional
        self.expand_all_condition_dotpaths()

        # Filter out the unconditional sweeps and then process them
        uncond_disjoint_sweeps = list(filter(compose(is_seq, 1),
                                             self.swept_disjoint_parameters))
        uncond_product_sweeps = list(filter(compose(is_seq, 1),
                                            self.swept_product_parameters))

        self.all_combinations = self.process_unconditional_sweeps(uncond_disjoint_sweeps,
                                                                  uncond_product_sweeps)

        # Filter out the conditional sweeps and then process them
        cond_disjoint_sweeps = list(filter(compose(is_mapping, 1), self.swept_disjoint_parameters))
        cond_product_sweeps = list(filter(compose(is_mapping, 1), self.swept_product_parameters))

        self.process_conditional_sweeps(cond_disjoint_sweeps,
                                        cond_product_sweeps)

        # Generate all the Quinfigs
        self.quinfigs = []
        for combination in self.all_combinations:
            coll = deepcopy(sweep_config)
            for parameter in combination:
                coll = tz.assoc_in(coll, parameter.path, parameter.value)
            self.quinfigs.append(Quinfig(config=coll, base_path=os.path.dirname(os.path.abspath(sweep_config_path))))

        print(f"Generated {len(self.quinfigs)} quinfig(s) successfully.")

    def __getitem__(self, idx):
        """
        Implement indexing into the QuinSweep to fetch particular Quinfigs.
        """
        return self.quinfigs[idx]

    def expand_all_condition_dotpaths(self):
        """
        Expands the paths
        """

        # Function that expands condition dotpaths for SweptParameters
        expand_condition_dotpaths = lambda sp: \
            SweptParameter(sp.path,
                           walk_values(lambda sweep: iffy(is_mapping,
                                                          autocurry(walk_keys)(
                                                              self.expand_reference))(sweep),
                                       sp[1])
                           )

        # Apply the function to the list of SweptParameters
        self.swept_parameters = \
            list(
                map(expand_condition_dotpaths,
                    self.swept_parameters)
            )

        # Function that expands condition dotpaths for Swept___Parameters
        expand_condition_dotpaths = lambda subtype, sp: \
            subtype(sp.path,
                    iffy(is_mapping,
                         autocurry(walk_keys)(
                             self.expand_reference))(sp[1]),
                    )
        expand_condition_dotpaths = autocurry(expand_condition_dotpaths)

        self.swept_disjoint_parameters = \
            list(

                map(expand_condition_dotpaths(SweptDisjointParameter),
                    self.swept_disjoint_parameters)
            )
        self.swept_product_parameters = \
            list(
                map(expand_condition_dotpaths(SweptProductParameter),
                    self.swept_product_parameters)

            )

        # Recreate the lookup table
        self.swept_parameters_dict = dict(zip(map(lambda sp: ".".join(sp.path),
                                                  self.swept_parameters),
                                              self.swept_parameters)
                                          )

    def replace_underscores(self, swept_parameter):
        """
        Replace all the underscore references in sweep of swept_parameter.
        """

        # Find all the references (i.e. dependencies) made by the swept_parameter
        references = []
        for token in QuinSweep.SWEEP_TOKENS[:-1]:  # omit default since it's value is never a dict
            if f"~{token}" in swept_parameter.sweep and is_mapping(swept_parameter.sweep[f"~{token}"]):
                references.extend(list(swept_parameter.sweep[f"~{token}"].keys()))

        # Find all the referred parameters
        parsed_references = list(map(QuinSweep.parse_ref_dotpath, references))
        dotpaths = list(cat(parsed_references))
        ref_dict = merge_with(compose(list, cat), *list(map(lambda e: dict([e]), dotpaths)))

        # TODO: there's a bug here potentially
        assert all(map(lambda l: len(l) == len(set(l)), list(itervalues(ref_dict)))), \
            'All conditions must be distinct.'

        ref_dict_no_underscores = walk_values(compose(set,
                                                      autocurry(map)(int),
                                                      autocurry(filter)(lambda e: e != '_')),
                                              ref_dict)

        if not references:
            return swept_parameter

        def compute_possibilities(full_dotpath, reference):
            # Look up the parameter using the dotpath
            parameter = self.swept_parameters_dict[full_dotpath]

            # Use the reference to figure out how many possiblities exist for the underscore
            if len(reference) > 0:
                # Merge all the sweeps performed for this parameter
                merged_sweep = merge(*list(filter(is_mapping, itervalues(parameter.sweep))))
                # Look up the reference
                return len(merged_sweep[reference])

            assert len(parameter.sweep) == 1, 'If no reference, must be a single unconditional sweep.'
            # The number of possibilities is simply the number of values specified
            # in the (product/disjoint) unconditional sweep
            return len(list(parameter.sweep.values())[0])

        # Update the sweep by replacing underscores
        updated_sweep = swept_parameter.sweep

        # Loop over all the parsed references
        for parsed_ref in parsed_references:

            # Expand all the partial dotpaths
            # TODO: remove? expanding all the partial dotpaths in the beginning?
            parsed_ref = list(
                map(lambda t: (self.expand_partial_dotpath(t[0]), t[1]), parsed_ref))

            # For each parsed reference, there will be multiple (dotpath, idx) pairs
            for i, (full_dotpath, ref_idx) in enumerate(parsed_ref):

                # If the reference index is not an underscore, continue
                if not ref_idx == '_':
                    continue

                # Compute the prefix reference
                prefix_reference = ".".join(list(cat(parsed_ref[:i])))

                # Compute the number of possible ways to replace the underscore
                n_possibilities = compute_possibilities(full_dotpath, prefix_reference)
                replacements = set(range(n_possibilities)) - ref_dict_no_underscores[full_dotpath]

                # Find the path to the underscore condition
                path_to_condition = get_only_paths(updated_sweep, lambda p: any(lambda e: '_' in e, p),
                                                   stop_at=full_dotpath)[0]

                # Find the value of the underscore condition
                value = tz.get_in(
                    path_to_condition,
                    updated_sweep)

                # Construct keys that are subtitutes for the underscore
                keys = list(map(lambda s: f'{full_dotpath}.{s}', replacements))
                keys = list(map(lambda k: path_to_condition[:-1] + [k], keys))

                # Update by adding those keys in
                for k in keys:
                    updated_sweep = tz.assoc_in(updated_sweep, k, value)

        # Create a new swept parameter with the updated sweep
        swept_parameter = SweptParameter(swept_parameter.path,
                                         walk_values(
                                             iffy(is_mapping,
                                                  autocurry(select_keys)(lambda k: '_' not in k)
                                                  ),
                                             updated_sweep)
                                         )
        return swept_parameter

    def process_conditional_sweeps(self,
                                   cond_disjoint_sweeps,
                                   cond_product_sweeps,
                                   ):
        """
        Function to process conditional sweeps: these sweeps are applied when a conditional is satisfied.

        As an example, consider

        """
        # The complete matrix of all pairwise comparisons over the set of conditional sweeps:
        # Note that our comparison op isn't transitive (e.g. A = B and B = C but A > C is possible)
        # so we want a topological sort over the set of sweeps: O(n^2) is incurred to generate the DAG
        # over which the toposort is applied
        all_comparisons = list(map(lambda t: (t, QuinSweep.param_comparator(*t)),
                                   itertools.product(cond_product_sweeps + cond_disjoint_sweeps,
                                                     cond_product_sweeps + cond_disjoint_sweeps)))

        dependencies = merge(
            # first pretend that there are no dependencies
            dict(map(lambda t: ((t.path, type(t)), set()), cond_disjoint_sweeps)),
            dict(map(lambda t: ((t.path, type(t)), set()), cond_product_sweeps)),
            # then merge in the dependencies
            dict(map(lambda t: ((t[0][0].path, type(t[0][0])), set([(t[0][1].path, type(t[0][1])), ])),
                     filter(lambda t: t[1] == -1, all_comparisons)
                     )
                 )
        )

        # Topological sort to produce the partial ordering: a list of sets for the partial ordering
        sweep_posets = list(toposort(dependencies))

        # Map to extract the dotpaths of the sweeps
        dotpath_posets = list(map(lambda s: set(map(0, s)), sweep_posets))

        # A dotpath could occur in more than one poset
        # e.g. when it contains both a product sweep and a disjoint sweep with different conditionals
        # Ensure that the dotpath occurs exactly once, in its earliest location
        dotpath_poset_subs = list(
            reversed(list(accumulate([set()] + list(reversed(dotpath_posets))[:-1], lambda a, b: a.union(b)))))
        dotpath_posets = list(map(lambda t: t[0] - t[1], zip(dotpath_posets, dotpath_poset_subs)))

        # Loop over all the sets in the partial order
        for poset in dotpath_posets:
            # For each path, convert it to a dotpath and then replace any underscores in it
            for path in poset:
                dotpath = QuinSweep.path_to_dotpath(path)
                self.swept_parameters_dict[dotpath] = self.replace_underscores(self.swept_parameters_dict[dotpath])

        for poset in sweep_posets:
            # Split the poset into the disjoint and product sweeps
            disjoint_poset = [p for p in poset if QuinSweep.is_disjoint_subtype(p[1])]
            product_poset = [p for p in poset if QuinSweep.is_product_subtype(p[1])]

            # Process the product sweeps first
            for path, subtype in product_poset:
                dotpath = QuinSweep.path_to_dotpath(path)
                # Each parameter combination needs to be expanded
                # print("Product Expansion")
                # for thing in self.all_combinations:
                #     print([e.dotpath for e in thing])
                self.all_combinations = self.product_expansion(self.all_combinations,
                                                               dotpath,
                                                               self.swept_parameters_dict[dotpath].sweep['~product'])

            # Process the poset
            # For disjoint, process the entire poset together (restricted to the disjoint sweeps)
            self.all_combinations = self.disjoint_expansion(self.all_combinations, disjoint_poset)

            pass

        # Apply the defaults
        for poset in dotpath_posets:
            # For each path, convert it to a dotpath and then replace any underscores in it
            for path in poset:
                for i, combo in enumerate(self.all_combinations):
                    dotpath = QuinSweep.path_to_dotpath(path)
                    if dotpath not in list(map(lambda c: self.path_to_dotpath(c.path), combo)):
                        combo.append(Parameter(path,
                                               f'{dotpath}.0',
                                               self.swept_parameters_dict[dotpath].sweep['~default']))
                        self.all_combinations[i] = combo

    def product_expansion(self,
                          combinations,
                          dotpath,
                          sweep):
        """
        Takes as input

        """
        new_combinations = []
        # Iterate over the parameter combinations
        for i, combo in enumerate(combinations):
            # print(i, combo, sweep)
            flag = False
            # The parameter combination contains a setting of the previously configured parameters
            for key in sweep:
                # First parse the key, which is a reference dotpath, to a list of dotpaths
                # Check if all the ref dotpaths are satisfied by the combo
                if all(map(lambda t: ".".join(t) in combo, self.parse_ref_dotpath(key))):
                    # Perform the product expansion
                    new_combos = list(itertools.product(*list(map(lambda e: [e], combo)),
                                                        list(
                                                            map(lambda e: Parameter(self.dotpath_to_path(dotpath),
                                                                                    f'{dotpath}.{e[0]}',
                                                                                    e[1]),
                                                                enumerate(sweep[key])
                                                                )
                                                        ))
                                      )
                    new_combos = list(map(compose(list, concat), new_combos))
                    new_combinations.extend(new_combos)
                    flag = True
                    break
            if not flag:
                new_combinations.append(combo)

        return new_combinations

    def disjoint_expansion(self,
                           combinations,
                           poset):
        """

        """

        ref_dotpath_to_dotpath = {}
        for path, _ in poset:
            # Look up the SweptParameter using the path
            swept_param = self.swept_parameters_dict[self.path_to_dotpath(path)]
            # Loop over the conditionals in the SweptParameter to create the lookup table
            for ref_dotpath in swept_param.sweep['~disjoint']:
                if ref_dotpath not in ref_dotpath_to_dotpath:
                    ref_dotpath_to_dotpath[ref_dotpath] = set()
                ref_dotpath_to_dotpath[ref_dotpath].add(swept_param)

        new_combinations = []

        # Iterate over the parameter combinations
        for i, combo in enumerate(combinations):
            # print(i, combo)
            flag = False
            # The parameter combination contains a setting of the previously configured parameters
            for ref_dotpath in ref_dotpath_to_dotpath:
                # First parse the key, which is a reference dotpath, to a list of dotpaths
                # Check if all the ref dotpaths are satisfied by the combo
                if all(map(lambda t: ".".join(t) in combo, self.parse_ref_dotpath(ref_dotpath))):
                    disjoint_parameters = list(zip(*map(lambda sp: list(map(lambda v: Parameter(sp.path,
                                                                                                f'{self.path_to_dotpath(sp.path)}.{v[0]}',
                                                                                                v[1]),
                                                                            enumerate(
                                                                                sp.sweep['~disjoint'][ref_dotpath])
                                                                            )
                                                                        ),
                                                        ref_dotpath_to_dotpath[ref_dotpath])
                                                   ))

                    new_combos = list(map(compose(list, cat), zip([combo] * len(disjoint_parameters),
                                                                  disjoint_parameters)
                                          )
                                      )
                    new_combinations.extend(new_combos)
                    flag = True
                    break

            if not flag:
                new_combinations.append(combo)

        return new_combinations

    def process_unconditional_sweeps(self,
                                     uncond_disjoint_sweeps,
                                     uncond_product_sweeps):
        """
        Parameters with unconditional sweeps are always processed first, and unconditional sweeps specify how to vary
        these parameters.

        Sweeps should be thought of as trees, with the root node containing assignments to all unswept parameters.
        Each time one or a group of swept parameters are processed, the tree grows in depth. The path from the root
        of the tree to any leaf indicates a particular assignment to the swept parameters.

        Unconditional sweeps can be

        - disjoint over r parameters, each parameter taking exactly k parameter values

        Unconditional disjoint sweeps can be thought of as generating k disjoint sweeps,
        where each sweep contains assignments to the r parameters.
        The k parameter values specified for each of the r parameters are assumed to be in alignment.

        In terms of the tree structure, this operation adds k children to the root node,
         with each child containing a simultaneous assignment to the r parameters.

        - product over p parameters, with the parameters taking k_1, k_2, ..., k_p parameter values

        Unconditional product sweeps generate k_i new sweeps for the ith (of p) parameter. Each of these product sweeps
        add a new layer to the tree, leading to a total depth increase of p. The ith such layer contains assignments
        to the ith parameter.

        In total, unconditional sweeps generate a total of k * k_1 * k_2 ... * k_r total leaf nodes in the tree.
        If the sweep only contains unconditional sweeps, the leaf nodes will contain complete assignments to all
        parameters, and can be used to create a Quinfig.
        """

        # All of the disjoint unconditional sweeeps must be identical length
        assert allequal(
            map(compose(len, 1), uncond_disjoint_sweeps)), 'All disjoint unconditional sweeps must have same length.'
        disjoint_uncond_combinations = list(zip(
            *map(
                lambda t: list(
                    map(lambda e: Parameter(t.path, f'{".".join(t.path)}.{e[0]}', e[1]), enumerate(t.disjoint))),
                uncond_disjoint_sweeps))
        )
        all_combinations = disjoint_uncond_combinations

        # Next, process the sequential product sweeps
        product_uncond_combinations = list(
            itertools.product(
                *map(lambda t: list(
                    map(lambda e: Parameter(t.path,
                                            f'{".".join(t.path)}.{e[0]}',
                                            e[1]),
                        enumerate(t.product)
                        )
                ),
                     uncond_product_sweeps)
            )
        )
        all_combinations = list(
            map(compose(list, cat),
                itertools.product(all_combinations,
                                  product_uncond_combinations)
                )
        )

        return all_combinations

    def fetch_swept_parameters(self,
                               config,
                               sweep_paths,
                               swept_parameter_paths):
        """
        Construct named tuples for all the parameters being swept.
        """
        # Filter sweep paths to all 3 sweep types
        disjoint_sweep_paths = list(filter(lambda s: 'disjoint' in last(s), sweep_paths))
        product_sweep_paths = list(filter(lambda s: 'product' in last(s), sweep_paths))
        default_sweep_paths = list(filter(lambda s: 'default' in last(s), sweep_paths))

        # Construct SweptParameter and Swept__Parameter namedtuples,
        # making it
        # consisting of the path to the parameter and its sweep configuration
        construct_swept_parameters = lambda subtype, paths, wrapper: list(
            map(lambda p: subtype(wrapper(p),
                                  tz.get_in(p, coll=config)
                                  ),
                paths)
        )

        swept_parameters = construct_swept_parameters(SweptParameter,
                                                      swept_parameter_paths,
                                                      identity)
        swept_disjoint_parameters = construct_swept_parameters(SweptDisjointParameter,
                                                               disjoint_sweep_paths,
                                                               QuinSweep.get_parameter_path)
        swept_product_parameters = construct_swept_parameters(SweptProductParameter,
                                                              product_sweep_paths,
                                                              QuinSweep.get_parameter_path)
        swept_default_parameters = construct_swept_parameters(SweptDefaultParameter,
                                                              default_sweep_paths,
                                                              QuinSweep.get_parameter_path)

        return swept_parameters, swept_disjoint_parameters, swept_product_parameters, swept_default_parameters


if __name__ == '__main__':
    import os

    print(os.getcwd())
    # sweep_config = Quinfig(
    #     config_path='quinine/tests/derived-1-2.yaml')
    sweep_config = Quinfig(
        config_path='tests/bugs/test-2.yaml')
    # sweep_config = Quinfig(
    #     config_path='/Users/krandiash/Desktop/workspace/projects/quinine/tests/derived-2.yaml')

    quin_sweep = QuinSweep(sweep_config=sweep_config)
    print(quin_sweep.quinfigs)
