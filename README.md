# Quinine
Quinine is a library to create and manage configuration files (called _quinfigs_).
It is especially well suited to machine learning projects, which typically specify a large number of hyperparameters.

**Quinine is designed to be extremely simple to use, with features rich enough for 
training and configuring machine learning models.**

## Features
Quinine is simple, powerful and extensible, 
providing an accessible window into the more complex and feature-rich gin configuration library. 

### Configuration in YAML
In the simplest case, just create a _Quinfig_ using a yaml file. Here's an example!

#### **`config.yaml`** 
```yaml 
general:
    seed: 2
    module: test.py

model:
    pretrained: true

dataset:
    - name: cifar10
    - name: imagenet 
```

#### **`main.py`** 
```python
# Import
from quinine import Quinfig

# Use the Quinfig class to create a quinfig 
quinfig = Quinfig(config_path='path/to/config.yaml')

# Access parameters as keys
assert quinfig['general']['seed'] == 2

# or use dot access, making your code cleaner
assert quinfig.general.seed == 2

# dot access works to arbitrary levels of nesting, including through lists
assert dataset[0].name == 'cifar10'

# you can also create a Quinfig directly from a dictionary
quinfig = Quinfig(config={'key': value})
```

### Templating for variation
A common use-case in machine learning is performs sweeps or variants on an experiment. It is convenient in this case to only have to specify the parameters that need to be changed from a base or template config. 


Quinine provides support for templating, using the special `templating` key in the config. Here's an example, where we specify a base configuration called `grandparent.yaml`, then inherit that while changing a single parameter in `parent.yaml`, and then inherit that while changing a single parameter in `config.yaml`. Recursive templating to arbitrary depth is supported!


#### **`grandparent.yaml`** 
```yaml 
general:
    seed: 2
    module: test.py

model:
    pretrained: true

dataset:
    - name: cifar10
    - name: imagenet 
```

#### **`parent.yaml`** 
```yaml 
templating:
    parent_yaml: path/to/grandparent.yaml 

# Overwrites the dataset configuration in grandparent.yaml to only train on CIFAR-10
dataset:
    - name: cifar10

# All other configuration options are inherited from grandparent.yaml
```

#### **`config.yaml`** 
```yaml 
templating:
    parent_yaml: path/to/parent.yaml 

# Overwrites part of the model configuration in parent.yaml (which equals its value in grandparent.yaml) to set pretrained to False
model:
	pretrained: false
```

#### **`main.py`** 
```python
# Nothing special needed: just create a quinfig normally
quinfig = Quinfig(config_path='path/to/config.yaml')

# and things will be resolved correctly
assert quinfig.model.pretrained == False
```


### Cerberus schemas for validation
Quinine uses Cerberus to support schema validation for your config files, and comes with syntactic sugar that will help you write most schemas very quickly. Schemas ensure your config is written correctly and can help flag when you (invariably) make mistakes. They're also useful if you want to mark parameters as required, specify defaults or parameter choices -- all the functionality available in Cerberus is supported, but you can just stick to simple use-cases that cover most scenarios with the syntatic sugar provided.

```python
# Write schemas in Python for reusability (recommended)

# The model schema contains a single 'pretrained' bool parameter that is required
model_schema = {'pretrained': merge(tboolean, required)}

# The schema for a single dataset contains its name
dataset_schema = {'name': tstring}

# The general schema consists of the seed (defaults to 0) and a module name (defaults to None)
general_schema = {'seed': merge(tinteger, default(0)), 'module': merge(tstring, nullable, default(None))}

# The overall schema is composed of these three reusable schemas
# Notice that you don't need to provide a schema for templating, Quinine will take care of that
schema = {'general': schema(general_schema), 'model': schema(model_schema), 'dataset': stlist(dataset_schema)}

# Just pass in the schema while instantiating the Quinfig: validation happens automatically
quinfig = Quinfig(config_path='path/to/config.yaml', schema=schema)

# You could also define schemas in YAML, but we recommend using Python to take advantage of the syntactic sugar
quinfig = Quinfig(config_path='path/to/config.yaml', schema_path='path/to/schema')
```



### Gin for sophisticated configuration
Gin is a feature-rich configuration library that gives users the ability to directly force a function argument in their code to take on some value. This can be especially useful when configuration files have nested dependencies: e.g. consider a config with an `optimizer` key that dictates which optimizer is built and used. Each optimizer (e.g. SGD or Adam) has its own configuration options (e.g. momentum for SGD or beta_1, beta_2 for Adam). With gin, you avoid having to create a schema that specifies every parameter for every possible optimizer in your config file (and/or writing boilerplate code to parse all of this).

Instead, you can mark functions as gin configurable (e.g. torch.optim.Adam and torch.optim.SGD) and simply set the arguments for the one you'll be using, directly in the config e.g. `torch.optim.Adam.beta_1 = 0.5`. No need to parse this gin configuration manually!

Quinine provides a thin wrapper on gin that allows users to perform gin configuration in YAML, without having to commit to gin completely (which can be cumbersome). With Quinine you can choose not to perform any gin configuration, use it a only a little or even use gin only, all from the convenience of YAML. Secondly, you can make your codebase gin configurable without having to manually decorate every function as `@gin.configurable`. This lets you switch away from gin without any hassles.



