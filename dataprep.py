import numpy as np
import pickle
import os
from random import randint
from sklearn.preprocessing import MinMaxScaler
from math import atan, sqrt, degrees


polar = True



train = {
    'val' : [],
    'key' : []
}
val = {
    'val' : [],
    'key' : []
}

scaler = MinMaxScaler()

def whatistrig(x,y):

    if x > 0 and y > 0:
        angle = atan(y/x)
    elif x < 0 and y < 0:
        angle = 180 - degrees(atan(y/x))
    elif x < 0  and y > 0:
        angle = 180 + degrees(atan(y/x))
    elif x > 0 and y < 0:
        angle = 360 + degrees(atan(y/x))
    elif x == 0 and y > 0:
        angle = 90
    elif x == 0 and y < 0:
        angle = 270
    elif y == 0 and x > 0:
        angle = 0
    elif y == 0 and x < 0:
        angle = 180

    #angle = degrees(np.arctan2(y,x))
    
    magnitude = sqrt((x*x) + (y*y))

    return [angle, magnitude]




def view(data):
    sense1 = []
    sense2 = []
    sense3 = []

    for i in range(len(data)-1):
        sense1.append(data[i][0])
        sense2.append(data[i][1])
        sense3.append(data[i][2])

    max_value = max(sense1)
    max_index = sense1.index(max_value)
    if max_index < 250:
        start = max_index - 50
    else:
        start = max_index - 250
    return abs(start)

haha = False

for fdir in os.listdir("all"):
    data = pickle.load(open('all/' + fdir, "rb"))
    if haha == False:
        scaler.fit(data[:-1])
        haha = True
    key = data[-1]
    x = key[0]
    y = key[1]
    if polar == True:
        key = whatistrig(x,y)
        
    
    vals = []
    index = view(data)
    if index > 18000:
        index = 17000
    scaler.transform(data[:-1])
    vals = np.array(data[index:index + 1000]).flatten()
    #print(len(vals))
    if len(vals) < 6000:
        print(len(data))
        print(index)
        print("------------------------")
    #print(np_vals.shape)
    """
    for i in data[:-1]:
        for c in i:
            vals.append(int(c))
    """
        

    if randint(0, 9) > 7:
        val['key'].append(key)
        val['val'].append(vals)
    else:
        train['key'].append(np.array(key))
        train['val'].append(vals)
    #print(key)
    #print(len(vals))
train['key'] = np.array(train['key'])
print(train['key'].shape)
train['val'] = np.array(train['val'])
print(train['val'].shape)
val['key'] = np.array(val['key'])
val['val'] = np.array(val['val'])

pickle.dump(train, open('train5.dat', 'wb'))
pickle.dump(val, open('val5.dat', 'wb'))