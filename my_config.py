def my_config(hp):
    a = hp.select(['a', 'b', 'c'], default='a')
    b = hp.select({'x': 1, 'y': 2}, name='b', default='x')