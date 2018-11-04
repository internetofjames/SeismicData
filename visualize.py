import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy
import scipy.io


data = pickle.load(open("test.dat", "rb"))

sense1 = []
sense2 = []
sense3 = []

for i in range(len(data)-1):
    sense1.append(data[i][0])
    sense2.append(data[i][1])
    sense3.append(data[i][2])


plt.plot(sense1, linewidth=0.25, color='r')
plt.plot(sense2, linewidth=0.25, color='g')
plt.plot(sense3, linewidth=0.25, color='b')
plt.show()