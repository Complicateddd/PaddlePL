import pandas as pd 

sigle_lists=[1,2,3]
down_result_pred=[1,2,'']

d={ 'NUM' : range(3),
    'C' : ['c1','c2','c3'],
    'D' : ['d1','d2','d3','d4'],
    'E' : ['e1','e2','e3'],
    'B' : ['b1','b2','b3']}

df = pd.DataFrame({'id': sigle_lists, 'content': down_result_pred})