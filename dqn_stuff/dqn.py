from keras.models import Model, load_model
from keras.layers import Conv2D, Dense, Flatten, Input, concatenate
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
import json

def main():

    # prep data
    filename = '0_shuffled.json'
    data, labels = load_data(filename)
    
    # make and train model
    train_model(data, labels)
    
    # Predict
    all_a1 = data['a1']
    all_a2 = data['a2']
    all_a3 = data['a3']
    all_a4 = data['a4']
    X_test = {'a1': all_a1[2], 'a2': all_a2[2], 'a3': all_a3[2], 'a4': all_a4[2]}
    
    model = load_model('./dqn1.h5')
    y_test_pred = model.predict(X_test)
    print('label should be '+str(labels[2]))
    
    a = input('asdf')

def load_data(filename):
    jdata = json.load(open(filename))
    n_states = len(jdata)
    #print(n_states)
    flen = len(jdata[0]['state_w_action_pair'])
    data = np.zeros((n_states,flen))
    labels = np.zeros((n_states,))
    for d in range(n_states):
        for e in range(flen):
            data[d][e] = jdata[d]['state_w_action_pair'][e]
        labels[d] = jdata[d]['reward']
    shaped_data1 = np.reshape(data[:,:2028], (n_states,13,13,12))
    shaped_data2 = np.reshape(data[:,2028:2197], (n_states,13,13,1))
    shaped_data3 = np.reshape(data[:,2197:2366], (n_states,13,13,1))
    shaped_data4 = np.reshape(data[:,2366:], (n_states,20)) # 2386
    #data_cat = [shaped_data1, shaped_data2, shaped_data3, shaped_data4]
    data_cat = {'a1': shaped_data1, 'a2': shaped_data2, 'a3': shaped_data3, 'a4': shaped_data4}
    return data_cat, labels

def train_model(data1, labels, lr=0.01, batch_size=5, num_epochs=20):
    
    #num_examples = data1[0].shape[0]
    num_examples = data1['a4'].shape[0]
    print(num_examples)
    cutoff = int(num_examples * 0.8)
    print(cutoff)
    #X_train = []
    all_a1 = data1['a1']
    all_a2 = data1['a2']
    all_a3 = data1['a3']
    all_a4 = data1['a4']
    X_train = {'a1': all_a1[:cutoff], 'a2': all_a2[:cutoff], 'a3': all_a3[:cutoff], 'a4': all_a4[:cutoff]}
    y_train = np.zeros((cutoff,))
    #X_valid = []
    X_valid = {'a1': all_a1[cutoff:], 'a2': all_a2[cutoff:], 'a3': all_a3[cutoff:], 'a4': all_a4[cutoff:]}
    leftover = num_examples-cutoff
    y_valid = np.zeros((leftover,))
    for e in range(cutoff):
        #xlist = []
        #for x in range(len(data1)):
        #    xlist.append(data1[x][e])
        #X_train.append(xlist)
        y_train[e] = labels[e]
    for e in range(cutoff, num_examples):
        #xlist = []
        #for x in range(len(data1)):
        #    xlist.append(data1[x][e])
        #X_valid.append(xlist)
        y_valid[e-cutoff] = labels[e]

    # Create DQN
    model = make_dqn()

    # Create computational graph
    adm = Adam(lr=lr)  # Adam instead of Stochastic Gradient Descent
    model.compile(loss='mean_squared_error', optimizer=adm, metrics=['accuracy'])

    #print(type(X_train))
    #print(len(X_train))
    #print(type(y_train))
    #print(X_train[0])
    #print(y_train[0])
    #import pdb
    #pdb.set_trace()
    # Training
    model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=num_epochs, batch_size=batch_size, verbose=2)

    # Save
    model.save('./dqn1.h5')

    
    # 35 w 4, 6th floor Tuesdays 10-11am
    
    
#def test_dqn():
    #model.test_data()

def make_dqn():
    # number of features for state info = 13*13*12
    a1 = Input(shape=(13, 13, 12), name='a1')
    output1 = cnn_part(a1)
    # number of features for bomb timer info = 13*13
    a2 = Input(shape=(13, 13, 1), name='a2')
    output2 = cnn_part(a2)
    # number of features for flame timer info = 13*13
    a3 = Input(shape=(13, 13, 1), name='a3')
    output3 = cnn_part(a3)
    # number of other features = 8
    a4 = Input(shape=(20,), name='a4')
    alist = [output1,output2,output3,a4]
    # Join the inputs
    a5 = concatenate(alist, axis=-1)
    # Hidden layer
    all_as = dense_part(a5, 100, 'sigmoid')
    # Output layer
    output_final = dense_part(all_as, 1, 'linear')
    model = Model(inputs=[a1,a2,a3,a4], outputs=output_final)
    return model

def cnn_part(a):
    x_dim = 3
    y_dim = 3
    overlap = 2
    weight_decay = 0.01
    net = Conv2D(x_dim, y_dim, strides=overlap, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(a)
    # remove channel dimension
    net = Flatten()(net)
    return net

def dense_part(a,nodenum,activationtype):
    net = Dense(units=nodenum, activation=activationtype)(a)
    return net

def convert_obs(obs):
    agent_state = featurize(obs[0])
    p = agent_state.astype(np.int8)
    q = p.flatten().tolist()
    # remove 9,8,5,3,2,1 from end
    q2 = q[0:-4]
    q2.pop(-5)
    q2.pop(-4)
    # find pos of you, teammate, enemy1, enemy2

    myx, myy = obs[0]["position"]
    enemy1x, enemy1y = obs[1]["position"]
    teammatex, teammatey = obs[2]["position"]
    enemy2x, enemy2y = obs[3]["position"]

    # manhattan distance to teammate
    q2.append(abs(myx - teammatex) + abs(myy - teammatey))
    q2.append(abs(myx - enemy1x) + abs(myy - enemy1y))
    q2.append(abs(myx - enemy2x) + abs(myy - enemy2y))
    q2.append(abs(teammatex - enemy1x) + abs(teammatey - enemy1y))
    q2.append(abs(teammatex - enemy2x) + abs(teammatey - enemy2y))

    # hot-one encode first 169 values then tack q2[169:] onto the end:
    q3 = []
    for ind in range(169):
        # no fog(5), agentdummy(10), both enemies get same value
        arr = [0] * 12
        val = q2[ind]
        # Item in constants.py
        if val in [0, 1, 2, 3, 4]:
            arr[val] = 1
        elif val in [6, 7, 8, 9]:
            arr[val - 1] = 1
        elif val in [11, 12, 13]:
            arr[val - 2] = 1
        elif val == 14:
            arr[10] = 1
        q3.append(arr)
    q4 = [item for sublist in q3 for item in sublist]
    q5 = [q4, q2[169:]]
    q6 = [int(item) for sublist in q5 for item in sublist]

    r = np.asarray(all_actions, dtype=np.int8)
    s = r.tolist()
    d = {
        "state": q6,
        "actions": s
    }
    return d

    
main()