# Basic Example

Here's a simple example to get you started:

{% code overflow="wrap" %}
```python
from hypster import HP, config


@config
def my_config(hp: HP):
    # Define your configuration parameters
    model_type = hp.select(['CNN', 'RNN'], default='CNN')

    # Conditional parameters
    if model_type == 'CNN':
        kernel_size = hp.int_input(5, min=3, max=9)
    elif model_type == 'RNN':
        cell_type = hp.select(['LSTM', 'GRU'], default='LSTM')

    # Common parameters
    learning_rate = hp.number_input(0.001)
    batch_size = hp.select([32, 64, 128], default=64)

# instantiate the configuration
results = my_config(values={'model_type': 'CNN', 'kernel_size' : 9})
```
{% endcode %}

`results` will contain a dictionary of variables from the instantiated config:

```
model_type : 'CNN'
kernel_size: 9
learning_rate : 0.001
batch_size : 64
```
