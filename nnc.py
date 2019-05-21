import keras
import tensorflow as tf
import h5py

from keras.models import load_model
import pandas as pd
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense


print("Where does the child live?\nPress 0 for Rural\nPress 1 for Urban")
a=int(input())

print("Are the parents together?\nPress 0 for No\nPress 1 for Yes")
ps=int(input())

print("Rank Mother's education from 0 to 10?")
medu=int(input())
medu=medu/10

print("Rank Father's education from 0 to 10?")
fedu=int(input())
fedu=fedu/10

print("Rank student's study time from 0 to 10?")
s=int(input())
s=s/10



p=[a,ps,medu,fedu,s]
er=[]
er.append(p)
er3=np.asarray(er)

#print(er3)

model=load_model('arthstugrade.h5')



z=model.predict(er3)
q=z[0][0]*100
q2=((q*10)-400)/2
print("The predicted marks for this student are-")
print(q2)