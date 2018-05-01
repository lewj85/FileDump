import json
import random
import numpy as np

jdata = json.load(open('0_prep.json'))
#print('opened file')
d = input('opened file')
random.shuffle(jdata)
#print('finished shuffling')
d = input('finished shuffling')
states = len(jdata)
print(states)
d = input('games')


f = open('0_shuffled.json', 'a+')
f.write('[')
for g in range(states):
        
    m = {
        "state_w_action_pair" : jdata[g]['state_w_action_pair'],
        "reward" : float(jdata[g]['reward'])
    }
    n = json.dumps(m)
    f.write(n)
    if g != states-1:
        f.write(',')
    #print(g, states)
    f.close()
    f = open('0_shuffled.json', 'a+')
f.write(']')
f.close()

d = input('asdf')
