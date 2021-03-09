# Quinine
Quinine is a no-nonsense, feature-rich library to create and manage configuration files (called _quinfigs_).
  
It's especially well suited to machine learning projects (designed by an ML PhD student @ Stanford aka me) where, 
- the number of hyperparameters can be quite large and are naturally nested
- projects are always expanding, so hyperparameters grow
- complicated manual hyperparameter sweeps are the norm 

## Installation
Install using pip,
```shell script
pip install quinine
```
For the latest version, 
```shell script
pip install git+https://github.com/krandiash/quinine.git --upgrade
```

## Features
Quinine is simple, powerful and extensible: let's go over all of the features with lots of examples. 

### Configuration in YAML
Configs are called Quinfigs. The most basic thing you can do is to create a _Quinfig_ using a yaml file. 

Here's an example where we use a `config.yaml` file to create a Quinfig. The only rule is you can't prefix any key
with the `~` character because we'll use that for sweeps.

#### **`config.yaml`** 
```yaml 
general:
    seed: 2
    module: test.py

model:
    pretrained: true
    ~architecture: resnet50 # <-- no ~ prefix allowed!
    architecture: resnet50 # <-- do this instead!

dataset:
    - name: cifar10
    - name: imagenet 
```

#### **`main.py`** 
```python
# Import the Quinfig class from quinine
from quinine import Quinfig

# Use the Quinfig class to create a quinfig 
# ...you can just pass in the path to the yaml 
quinfig = Quinfig(config_path='path/to/config.yaml')

# Access parameters as keys
assert quinfig['general']['seed'] == 2

# or use dot access, making your code cleaner
assert quinfig.general.seed == 2

# dot access works to arbitrary levels of nesting, including through lists
assert quinfig.dataset[0].name == 'cifar10'

# you can also create a Quinfig directly from a dictionary
quinfig = Quinfig(config={'key': 'value'})
```

YAMLs are great for writing large, nested configs cleanly, and provide a nice separation from your code. This configuration workflow 
(feed yaml to python script) is pretty popular, and if all you wanted was that, Quinine has you covered. 

Read on to see more! 

### Inheritance for configs
A common use-case in machine learning is performing sweeps or variants on an experiment. 
It's often convenient to have to specify _only_ the parameters that need to be changed from some 'base' or template configs. 


Quinine provides support for inheritance, using the special `inherit` key in the config. 

Here's an example, where we 
- first specify a base config called `grandparent.yaml`,
- inherit this config in `parent.yaml` and change a single parameter,
- then inherit _that_ in `config.yaml`, changing another parameter. 


#### **`grandparent.yaml`** 
```yaml 
general:
    seed: 2
    module: test.py

model:
    pretrained: true
    architecture: resnet50

dataset:
    - name: cifar10
    - name: imagenet 
```

#### **`parent.yaml (how you write it)`** 
```yaml 
inherit: path/to/grandparent.yaml 

# Overwrites the dataset configuration in grandparent.yaml to only train on CIFAR-10
dataset:
    - name: cifar10

# All other configuration options are inherited from grandparent.yaml
```

#### **`parent.yaml (how it actually is)`** 
```yaml 
inherit: path/to/grandparent.yaml 

general:
    seed: 2
    module: test.py

model:
    pretrained: true
    architecture: resnet50

dataset:
    - name: cifar10
```

#### **`config.yaml (how you write it)`** 
```yaml 
inherit: path/to/parent.yaml 

# Overwrites the model configuration in parent.yaml (which equals its value in grandparent.yaml) to set pretrained to False
model: 
    pretrained: false

# All other configuration options are inherited from parent.yaml
```

#### **`config.yaml (how it actually is)`** 
```yaml 
inherit: path/to/parent.yaml 

general:
    seed: 2
    module: test.py

model:
    pretrained: false
    architecture: resnet50

dataset:
    - name: cifar10
```

#### **`main.py`** 
```python
# Nothing special needed: just create a quinfig normally
quinfig = Quinfig(config_path='path/to/config.yaml')

# and things will be resolved correctly
assert quinfig.model.pretrained == False
assert quinfig.model.architecture == 'resnet50'
```

You can also inherit from multiple configs simultaneously (later configs take precedence). Here's an example,

#### **`config.yaml`**
```yaml
inherit: 
    - path/to/parent_1.yaml
    - path/to/parent_2.yaml
    - path/to/parent_3.yaml # later parameters take precedence  

general:
    seed: 2
    module: test.py

model:
    pretrained: false
    architecture: resnet50

dataset:
    - name: cifar10
```


### Cerberus schemas for validation
A nice-to-have feature is the ability to validate your config file against a schema.
 
If you've used `argparse` to ever configure your scripts, you've been doing this already. In a nutshell,
the schema lets you specify what hyperparameters the program will accept and if you pass in something that's
unexpected (e.g. architectur instead of architecture), it'll catch the error (that's called _schema validation_).

Quinine uses an external library called `Cerberus` to support schema validation for your config files.
Cerberus is great, but it has a bit of a learning curve and a lot of features you'll never actually use. 
So to make things easy, Quinine comes with syntactic sugar that will help you write schemas very quickly.
All the functionality available in Cerberus is supported, 
but most scenarios are covered with the syntatic sugar provided. 

Another reason to use schemas: you can mark parameters as required, specify defaults or choices for the parameter's values.
 
-- 

```python
from quinine import Quinfig, tstring, tboolean, tinteger, stdict, stlist, default, nullable, required
from funcy import merge
# You should write schemas in Python for reusability (recommended)

# The model schema contains a single 'pretrained' bool parameter that is required
model_schema = {'pretrained': merge(tboolean, required)}

# The schema for a single dataset contains its name
dataset_schema = {'name': tstring}

# The general schema consists of the seed (defaults to 0) and a module name (defaults to None)
general_schema = {'seed': merge(tinteger, default(0)), 
                  'module': merge(tstring, nullable, default(None))}

# The overall schema is composed of these three reusable schemas
# Notice that you don't need to provide a schema for templating, Quinine will take care of that
schema = {'general': stdict(general_schema), 
          'model': stdict(model_schema), 
          'dataset': stlist(dataset_schema)}

# Just pass in the schema while instantiating the Quinfig: validation happens automatically
quinfig = Quinfig(config_path='path/to/config.yaml', schema=schema)

# You could also define schemas in YAML, but we recommend using Python to take advantage of the syntactic sugar
quinfig = Quinfig(config_path='path/to/config.yaml', schema_path='path/to/schema')
```

### QuinineArgumentParser: Override Command-Line Arguments
Quinine also comes with an argument parser that can be used to perform command-line
 overrides on top of arguments specified in a config `.yaml` file.
 
 ```python
from quinine import QuinineArgumentParser
parser = QuinineArgumentParser(schema=your_schema) # a schema is necessary if you want to override command-line arguments
quinfig = parser.parse_quinfig()
# Do stuff
``` 

To use this, you can run
```shell script
# Load config from `your_config.yaml` and override `nested_arg.nesting.parameter` with
# a new value = 'abc'
> python your_file.py --config your_config.yaml --nested_arg.nesting.parameter abc
# ...and so on
> python your_file.py --config your_config.yaml --arg1 2 --arg2 'abc' --nested.arg a
```

Note that `your_config.yaml` can inherit from an arbitrary number of configs.

### QuinSweeps: YAML Sweeping on Steroids
Quinine has a _very_ powerful syntax for sweeps. One of the problems this aims to address is that
it's often convenient to write sweeps in Python, because you can use operations such as products, zips and chains. 
But it's ugly and cumbersome to manage parameters in Python and I personally like having the separation that YAML provides.

With Quinine, you can write complex sweeps with nested logic without leaving the comfort of your YAML file. 

Quinine will not actually run or manage your swept runs or do 'smart' hyperparameter optimization (hyperband-style). 

We'll go through a few examples to see how this works.

Scenario: sweep over 4 learning rates
```yaml
# This YAML specifies fixed values for all but one parameter: 
# optimizer.learning_rate takes on 4 values.
model:
    pretrained: false
    architecture: resnet50

optimizer:
    learning_rate: 
        # Sweep over 4 separate learning rates
        ~disjoint: # you could also have used the ~product key here -- note the use of the special ~ character
            - 0.01
            - 0.001
            - 0.0001
            - 0.00001
    scheduler: cosine
```

```python
from quinine import QuinSweep

# Generate a QuinSweep using this YAML
quinsweep = QuinSweep(sweep_config_path='path/to/sweep_config.yaml')

# Index into the quinsweep to get the i^th Quinfig
i = 3
quinfig_3 = quinsweep[3] # quinfig_i sets learning_rate to 0.00001 

# Iterate over the quinsweep
for quinfig in quinsweep:
    # Do something with the quinfig (e.g. run a job)
    your_fn_that_does_something(quinfig)
```

Scenario: sweep over 4 distinct parameter settings that specify learning rate and architecture
```yaml
model:
    pretrained: false
    architecture:
        # Sweep over 4 separate architectures
        ~disjoint: 
            - resnet18
            - resnet50
            - vgg19
            - inceptionv3

optimizer:
    learning_rate: 
        # Sweep over 4 separate learning rates
        ~disjoint:
            - 0.01
            - 0.001
            - 0.0001
            - 0.00001
    scheduler: cosine
```

Scenario: sweep over all possible combinations of 4 learning rates and 4 architectures
```yaml
model:
    pretrained: false
    architecture:
        # Sweep over 4 separate learning rates 
        ~product: 
            - resnet18
            - resnet50
            - vgg19
            - inceptionv3

optimizer:
    learning_rate: 
        # Sweep over 4 separate learning rates
        ~product:
            - 0.01
            - 0.001
            - 0.0001
            - 0.00001
    scheduler: cosine
```

Scenario: sweep over all possible combinations of 4 learning rates and 4 architectures and if architecture is resnet50,
additionally sweep over 2 learning rate schedulers
```yaml
model:
    pretrained: false
    architecture:
        # Sweep over 4 separate learning rates 
        ~product: 
            - resnet18
            - resnet50
            - vgg19
            - inceptionv3

optimizer:
    learning_rate: 
        # Sweep over 4 separate learning rates
        ~product:
            - 0.01
            - 0.001
            - 0.0001
            - 0.00001
    scheduler:
        # By default use the cosine scheduler
        ~default: cosine 
        ~disjoint:
            # But, when architecture takes on index 1 (i.e. resnet50), sweep over 2 parameters
            architecture.1: 
                - cosine
                - linear
```

### Gin for sophisticated configuration
`Gin` is a feature-rich configuration library that gives users the ability to directly force a function argument 
in their code to take on some value. 

This can be especially useful when configuration files have nested dependencies: 
e.g. consider a config with an `optimizer` key that dictates which optimizer is built and used. 
Each optimizer (e.g. SGD or Adam) has its own configuration options (e.g. momentum for SGD or beta_1, beta_2 for Adam).
 
With gin, you avoid having to create a schema that specifies every parameter for every possible optimizer in your 
config file (and/or writing boilerplate code to parse all of this).

Instead, you can mark functions as gin configurable (e.g. torch.optim.Adam and torch.optim.SGD) and 
simply set the arguments for the one you'll be using, directly in the config e.g. `torch.optim.Adam.beta_1 = 0.5`. 
When you need to use the optimizer, just use `torch.optim.Adam()` (and gin will take care of specifying the parameters).
No need to parse this gin configuration manually!

Quinine provides a thin wrapper on gin that allows users to perform gin configuration in YAML, 
without having to commit to gin completely (which can be cumbersome). 

With Quinine you can choose not to perform any gin configuration, use it a only a little or even use gin only, 
all from the convenience of YAML. 

Secondly, you can make your codebase gin configurable without having to manually decorate every function as `@gin.configurable`. 
This lets you switch to/away from gin without any hassles.


### About
If you use `quinine` in a research paper, please use the following BibTeX entry
```
@misc{Goel2021,
  author = {Karan Goel},
  title = {Quinine: Configuration for Machine Learning Projects},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/krandiash/quinine}},
}
```

### Acknowledgments
Thanks to Tri Dao and Albert Gu for initial discussions that led to the development
 of `quinine`, as well as Kabir Goel, Shreya Rajpal, Laurel Orr and Sidd
  Karamcheti for providing valuable feedback.
