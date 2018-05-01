import json
import random
import numpy as np

jdata = json.load(open('0.json'))
print('opened file')

games = len(jdata)
print(games)

f = open('0_prep.json', 'a+')
f.write('[')
for g in range(games):

    states_in_current_game = len(jdata[g]['states'])
    
    # create labels
    diminishing_reward_value = 0.9
    labels = diminishing_reward_value ** np.arange(states_in_current_game)
    labels *= int(jdata[g]['reward'])
    
    # assign labels, create dict, dump to file
    for j in range(states_in_current_game):
        
        a = jdata[g]['states'][j]['state']
        
        # add teammate's action
        b = [0,0,0,0,0,0]
        b[jdata[g]['states'][j]['actions'][0]] = 1
        
        # add our action
        c = [0,0,0,0,0,0]
        c[jdata[g]['states'][j]['actions'][2]] = 1
        
        d = [a, b, c]
        e = [item for sublist in d for item in sublist]
        #f.write(str(len(e)))
        m = {
            "state_w_action_pair" : e,
            "reward" : labels[j]
        }
        n = json.dumps(m)
        f.write(n)
        if j != states_in_current_game-1:
            f.write(',')
    if g != games-1:
        f.write(',')
    print(g, games)
    f.close()
    f = open('0_prep.json', 'a+')
f.write(']')
f.close()
