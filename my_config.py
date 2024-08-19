def my_config(hp: HP):
    a = hp.select(['a', 'b', 'c'], default='a')
    b = hp.select({'x': 1, 'y': 2}, name='b', default='x')
    c = hp.text_input('ette')
    conf = {'a': a}
    d = hp.number_input(4)