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

The function definition consists of the `@config` decorator and the signature. Including the `HP` (HyperParameter) type hint will enable IDE features like code suggestions and type checking.&#x20;
{% endstep %}

{% step %}
### Body

```python
@config
def my_config(hp: HP):
    from package import Class
    
    var = hp.select(["a", "b", "c"], default="a")
    num = hp.number(10)
    text = hp.text("Hey!")
    
    instance = Class(var=var, num=num, text=text)
```

Hypster comes with the following options:

* `hp.select()` and `hp.multi_select()` for categorical choices
* `hp.int()` and `hp.multi_int()` for integer values
* `hp.number()`and `hp.multi_number()` for numeric values
* `hp.text()` and `hp.multi_text()` for string values
* `hp.bool()` and `hp.multi_bool()` for boolean values

Please note:

{% hint style="warning" %}
**All imports must be defined inside the body of the function.** This enables the portability of hypster's configuration object.
{% endhint %}

{% hint style="info" %}
**No return statement is allowed (nor needed)**. This enables selecting the variables we want to retrieve upon instantiation using `final_vars` and `exclude_vars`
{% endhint %}
{% endstep %}

{% step %}
### Instantiation

Now that we've created a configuration space/function - we can instantiate it using:

```python
my_config(final_vars=["instance"], values={"var" : "b"})
```
{% endstep %}
{% endstepper %}

Congratulations! :tada: You've created and instantiated your first Hypster config.

To dive deeper into instantiation, please visit the [next section](instantiating-a-configuration-space.md).
