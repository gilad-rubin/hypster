# 🚀 Defining of A Config Function

A hypster `config` function is the heart of this framework. It requires of 3 main components:

{% stepper %}
{% step %}
### Imports

```python
from hypster import config, HP
```

This makes sure we have the `@config` decorater and `HP` class for type annotation.
{% endstep %}

{% step %}
### Signature

```python
@config
def my_config(hp: HP):
```

The function definition consists of the `@config` decorator and the signature. Including the `HP` (HyperParameter) type hint will enable IDE features like code suggestions and type checking.

#### Optional Registration

You can optionally register your configuration in the global registry for easy reuse:

```python
@config(register="models.my_model")
def my_config(hp: HP):
    # Configuration body...
```

The `register` parameter allows you to:
- **Store** configurations in a global registry with namespace organization
- **Reuse** configurations across different parts of your application
- **Discover** available configurations through `hp.nest()` calls

Registered configurations can be accessed using `hp.nest("models.my_model")` from any other configuration. See the [Configuration Registry](../advanced/registry.md) documentation for more details.
{% endstep %}

{% step %}
### Body

```python
@config
def my_config(hp: HP):
    from package import Class

    var = hp.select(["a", "b", "c"], default="a", name="var")
    num = hp.number(10, name="num")
    text = hp.text("Hey!", name="text")

    instance = Class(var=var, num=num, text=text)
```

Hypster comes with the following HP calls:

* `hp.select()` and `hp.multi_select()` for [categorical choices](../in-depth/hp-call-types/select-and-multi-select.md)
* `hp.int()` and `hp.multi_int()` for [integer values](../in-depth/hp-call-types/int-and-multi-int.md)
* `hp.number()`and `hp.multi_number()` for [numeric values](../in-depth/hp-call-types/int-and-multi-int.md)
* `hp.text()` and `hp.multi_text()` for [string values](../in-depth/hp-call-types/text-and-multi-text.md)
* `hp.bool()` and `hp.multi_bool()` for [boolean values](../in-depth/hp-call-types/bool-and-multi-bool.md)

**Important**: All HP method calls require a `name` parameter to identify the parameter.

Please note:

{% hint style="warning" %}
**All imports must be defined inside the body of the function.** This enables the portability of hypster's configuration object.
{% endhint %}

{% hint style="info" %}
**No return statement is allowed (nor needed)**. This enables [selecting the variables](selecting-output-variables.md) we want to retrieve upon instantiation using `final_vars` and `exclude_vars`
{% endhint %}
{% endstep %}

{% step %}
### Instantiation

Now that we've created a configuration space/function - we can instantiate it using:

```python
my_config(final_vars=["instance"], values={"var" : "b"})
```

Congratulations! :tada: You've created and instantiated your first Hypster config.
{% endstep %}
{% endstepper %}

## Saving & Loading Config Functions

Save configurations to reuse them across projects:

```python
# Save directly from config function
my_config.save("configs/my_config.py") # Creates directories if needed

# Save using hypster.save
from hypster import save
save(my_config, "configs/nested/my_config.py")
```

#### Loading Configurations

Load saved configurations in two ways:

```python
# Method 1: Direct loading
from hypster import load
my_config = load("configs/my_config.py")

# Method 2: Load for nesting
@config
def parent_config(hp: HP):
    nested_config = hp.nest("configs/my_config.py")
```
