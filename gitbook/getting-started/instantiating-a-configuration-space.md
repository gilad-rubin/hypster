# Instantiating a Config Function

<figure><img src="../.gitbook/assets/image (19).png" alt=""><figcaption></figcaption></figure>

## Configuration Function

In this section, we'll use the following toy configuration function:

```python
from hypster import config, HP

@config
def llm_config(hp: HP):
    model_name = hp.select({"sonnet" : "claude-3-5-sonnet-20241022"
                            "haiku" : "claude-3-5-haiku-20241022"}, 
                            default="haiku")
    
    if model_type == "haiku":
        max_tokens = hp.int(256, max=2048)
    else:
        max_tokens = hp.int(126, max=1024)
        
    cache = Cache(folder=hp.text("./cache"))
    config_dct = {"temperature" : hp.number(0, min=0, max=1),
                  "max_tokens" : max_tokens}
              
    model = Model(model_name, cache)
```

## Basic Parameter Instantiation

Parameter values can be set using a `values` dictionary:

```python
config1 = llm_config(values={"model_name" : "sonnet"})
config2 = llm_config(values={"max_tokens" : 256, "config_dct.temperature" : 0.5})
```

Key considerations:&#x20;

* Parameter values must align with the **conditional logic** in the configuration
* **Dot notation** (e.g. `config_dct.temperature`) is used for automatic naming of nested parameter. More on that in the [Automatic Naming](../in-depth/automatic-naming.md) section.
* **Default values** are used when specific values aren’t provided

## Instantiating `select` and `multi_select`

we can define hp.select or multi\_select using a list or a dictionary. a list is equal to {item: item} dictionary.

when you want to instantiate with select you either provide the key from the dictionary or define a value. the value can be outside of the predefined values.&#x20;

just note that if you provide a value that's not string, float, int, bool - hypster has a history mechanism for values and it will be saved as str(value). this means it won't be reproducible.&#x20;

example:

my\_config(values={"model\_name" : "claude-3-opus-20240229"}) will work and be serialized etc...

my\_config(values={"model\_name" : Complex(a=5)}) will also work, but won't be reproducible.

## Instantiating number & int (also multi\_number and multi\_int)

here it'll work for int - ints. for number - float or int works. just if you provide bounds (min or max) - it'll validate that.

## text or bool

will just make sure it's str or bool.

## instantiating hp.propagte

in order to instantiate it - it's nested config functions. we can use either dot notation or dictionary.

example:

```python
from hypster import config, HP

@config
def rag_config(hp: HP):
    llm = hp.propagate("path_to_llm_config.py")
    retriever = hp.propagate("path_to_retriever_config.py")
    
    pipeline = [retriever["embedder"], llm["model"]]
    #you can make this a bit better :)
```

so we can instantiate it via:

```python
rag_config(values={"retriever.embedding" : "jina", 
                   "llm" : {"model_name" : "haiku"}
                   })
```

