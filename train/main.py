import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
from SampleData import *

housedata=fetch_california_housing()
X_train,X_test,y_train,y_test,z_train,z_test=getData(0.8)
# scale=StandardScaler()
# X_train_s=scale.fit_transform(X_train)
# X_test_s=scale.fit_transform(X_test)

print(np.average((z_test - y_test) / z_test))

print(np.where(np.isnan(X_train)))
print(np.where(np.isnan(y_train)))

train_xt=torch.from_numpy(X_train.astype(np.float32))
train_yt=torch.from_numpy(y_train.astype(np.float32))
test_xt=torch.from_numpy(X_test.astype(np.float32))
test_yt=torch.from_numpy(y_test.astype(np.float32))

train_data=Data.TensorDataset(train_xt,train_yt)
test_data=Data.TensorDataset(test_xt,test_yt)
train_loader=Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=0)

class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression,self).__init__()
        self.hidden1=nn.Linear(in_features=12,out_features=128,bias=True)
        self.hidden2=nn.Linear(128,128)
        self.hidden3=nn.Linear(128,64)
        self.predict=nn.Linear(64,1)
    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        x=F.relu(self.hidden3(x))
        output=self.predict(x)
        return output[:,0]

mlpreg=MLPregression()
print(mlpreg)

optimizer=torch.optim.SGD(mlpreg.parameters(),lr=0.001)
loss_func=nn.MSELoss()
train_loss_all=[]
for epoch in range(30):
    train_loss=0
    train_num=0
    for step,(b_x,b_y) in enumerate(train_loader):
        output=mlpreg(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*b_x.size(0)
        train_num+=b_x.size(0)
    train_loss_all.append(train_loss/train_num)
plt.figure(figsize=(10,6))
plt.plot(train_loss_all,"ro-",label="Train loss")
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("train.jpg", format="jpg")
plt.show()

pre_y=mlpreg(test_xt)
pre_y=pre_y.data.numpy()
print(np.argwhere(np.isnan(np.array(test_xt))))
print(np.argwhere(np.isnan(pre_y)))
mae=np.average(np.abs(y_test-pre_y))
print(mae)
mape=np.average(np.abs(y_test-pre_y) / pre_y)
print(mape)

index=np.argsort(y_test)
plt.figure(figsize=(12,5))
plt.plot(np.arange(len(y_test)),y_test[index],"r",label="original y")
plt.scatter(np.arange(len(y_test)),z_test[index],s=3,c="g",label="2 time-addition")
plt.scatter(np.arange(len(pre_y)),pre_y[index],s=3,c="b",label="prediction")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel("index")
plt.ylabel("y")
plt.savefig("test.jpg", format="jpg")
plt.show()