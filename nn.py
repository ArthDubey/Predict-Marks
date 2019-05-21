import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense

data=pd.read_csv('student-mat.csv')
#print(data.head())
data["Medu"]=data['Medu'].apply(lambda x:x/4)
data["Fedu"]=data['Fedu'].apply(lambda x:x/4)
maxabs=max(data['absences'])
data["absences"]=data['absences'].apply(lambda x:x/maxabs)

maxabs=max(data['goout'])
data["goout"]=data['goout'].apply(lambda x:x/maxabs)


maxabs=max(data['studytime'])
data["studytime"]=data['studytime'].apply(lambda x:x/maxabs)

maxabs=max(data['failures'])
data["failures"]=data['failures'].apply(lambda x:x/maxabs)

maxabs=max(data['traveltime'])
data["traveltime"]=data['traveltime'].apply(lambda x:x/maxabs)

#maxabs=max(data['G1'])
data["G1"]=data['G1'].apply(lambda x:x/20)

#maxabs=max(data['G2'])
data["G2"]=data['G2'].apply(lambda x:x/20)

#maxabs=max(data['G3'])
data["G3"]=data['G3'].apply(lambda x:x/20.5)


data["Dalc"]=data['Dalc'].apply(lambda x:x/5)

data["Walc"]=data['Walc'].apply(lambda x:x/5)

data["health"]=data['health'].apply(lambda x:x/5)

data["famrel"]=data['famrel'].apply(lambda x:x/5)


print(max(data['goout']))
print(max(data['failures']))
print(max(data['traveltime']))

for index, row in data.iterrows():
    if(row['sex']=='F'):
        data.at[index,'sex']=0
    if(row['sex']=='M'):
        row['sex']=1
        data.at[index,'sex']=1

    if(row['address']=='U'):
        data.at[index,'address']=1
    if(row['address']=='R'):
        data.at[index,'address']=0

    if(row['Pstatus']=='A'):
        data.at[index,'Pstatus']=0
    if(row['Pstatus']=='T'):
        data.at[index,'Pstatus']=1

    if(row['guardian']=='mother'):
        data.at[index,'guardian']=0
    if(row['guardian']=='father'):
        data.at[index,'guardian']=1
    if(row['guardian']=='other'):
        data.at[index,'guardian']=0.5   
        

    if(row['nursery']=='yes'):
        data.at[index,'nursery']=1
    if(row['nursery']=='no'):
        data.at[index,'nursery']=0

    if(row['higher']=='yes'):
        data.at[index,'higher']=1
    if(row['higher']=='no'):
        data.at[index,'higher']=0

    if(row['internet']=='yes'):
        data.at[index,'internet']=1
    if(row['internet']=='no'):
        data.at[index,'internet']=0

    if(row['romantic']=='yes'):
        data.at[index,'romantic']=1
    if(row['romantic']=='no'):
        data.at[index,'romantic']=0     

data=data.drop(columns="school")
data=data.drop(columns="age")
data=data.drop(columns="famsize")
data=data.drop(columns="Fjob")
data=data.drop(columns="Mjob")
data=data.drop(columns="reason")
data=data.drop(columns="schoolsup")
data=data.drop(columns="famsup")
data=data.drop(columns="paid")
data=data.drop(columns="activities")
data=data.drop(columns="freetime")
data=data.drop(columns="G1")
data=data.drop(columns="G2")
data=data.drop(columns="traveltime")
data=data.drop(columns="sex")
data=data.drop(columns="guardian")
data=data.drop(columns="famrel")
data=data.drop(columns="goout")
data=data.drop(columns="Dalc")

data=data.drop(columns="Walc")
data=data.drop(columns="health")
data=data.drop(columns="absences")


data=data.drop(columns="failures")
data=data.drop(columns="nursery")
data=data.drop(columns="higher")
data=data.drop(columns="internet")
data=data.drop(columns="romantic")





df1=data.iloc[0:387,0:5]
df2=data.iloc[0:387,5:6]
print(df2)

dftest1=data.iloc[387:390,0:5]
dftest2=data.iloc[387:390,5:6]
print(dftest2)

#print(df1)
dfa1=np.asarray(df1)
dfa2=np.asarray(df2)
dfta1=np.asarray(dftest1)
dfta2=np.asarray(dftest2)



model = Sequential([
                    Dense(32, activation='relu', input_shape=(5,)),
                    Dense(64, activation='relu'),
            
                    Dense(32, activation='relu'),

                    Dense(1, activation='sigmoid'),
                    ])



sgd = optimizers.Adam(lr=0.001)
model.compile(optimizer='sgd',
                loss='binary_crossentropy',
              metrics=['accuracy'])
print(data)
model.fit(dfa1,dfa2,epochs=500)


test_loss, test_acc = model.evaluate(dfta1, dfta2)

print('Test accuracy:', test_acc)
#model.save('arthstugrade.h5')

