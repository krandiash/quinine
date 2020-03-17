from common.cerberus import *
from common.gin import *
import examples.simple
from quinfig import Quinfig
from examples.simple import simple_program


def simple_example():
    # Create a simple schema using Cerberus: shortcuts make it easy to write complex schemas with reusable components
    schema = {'general': stdict({'seed': merge(tinteger, default(0)),
                                 'module': merge(tstring, required),
                                 }),
              'model': stdict({'architecture': merge(tstring, allowed(['resnet18', 'resnet50'])),
                               'pretrained': merge(tboolean, required)
                               }),
              }

    # Internally, Quinfig autoupdates the schema using the autoexpand_schema function to support gin
    prettyprint(autoexpand_schema(schema))

    # Write out the config: you could also have written this in a yaml file
    config = {'general': {'seed': 2,
                          'module': 'test.py'},
              'model': {'pretrained': True},
              'gin':
              # set the print_yes argument in a_gin_configurable_fn to True
              # you can use as much or as little gin configuration as you like
              # e.g. you could write your entire configuration in gin or not use gin at all
                  {'a_gin_configurable_fn.print_yes': True},
              'templating':
              # first inherit all the configuration settings in tests/base.yaml and then overwrite them with this config
                  {'parent_yaml': 'examples/base.yaml'}
              }

    # Register the module that we want to configure with gin
    register_module_with_gin(examples.simple, 'examples.simple')

    # Create the quinfig
    quinfig = Quinfig(config=config,
                      schema=schema)
    # Voila!
    simple_program(quinfig)

    # Or you could have used the yaml
    quinfig = Quinfig(config_path='examples/config.yaml',
                      schema=schema)
    simple_program(quinfig)


if __name__ == '__main__':
    simple_example()
