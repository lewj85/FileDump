import json
import random
import numpy as np

jdata = json.load(open('yichen_clean1.json'))
print('opened file')

games = len(jdata)
print(games)

f = open('yichen_simple_1.json', 'a+')
f.write('[')
for g in range(games):

    num_states_in_current_game = len(jdata[g]['states'])
    
    s = []
    # assign labels, create dict, dump to file
    for j in range(num_states_in_current_game):
        
        a = jdata[g]['states'][j]['state']
        
        # add teammate's action
        b = [0,0,0,0,0,0]
        b[jdata[g]['states'][j]['actions'][0]] = 1
        
        # add our action
        c = [0,0,0,0,0,0]
        c[jdata[g]['states'][j]['actions'][2]] = 1
        
        d = [a, b, c]
        e = [item for sublist in d for item in sublist]

        s.append(e)
        
    m = {
        "states_w_action_pairs" : s,
        "reward" : jdata[g]['reward']
    }
    n = json.dumps(m)
    f.write(n)
    if g != games-1:
        f.write(',')
    #print(g, games)
    f.close()
    f = open('yichen_simple_1.json', 'a+')
f.write(']')
f.close()
