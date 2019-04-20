#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[17]:




print("Enter No of hidden nodes")
hidnodes=int(input())
print("ENter learning rate")
learningrate=float(input())
import csv

import numpy as np


datapath = "/home/ajit/Desktop/ML/ell-409-assignment-1/"
# /home/ajit/Downloads/ell409demo1
# train.csv
traindata = np.loadtxt(datapath + "mnist_train.csv", 
                        delimiter=",")
testdata = np.loadtxt(datapath + "mnist_test.csv", 
                       delimiter=",") 

factor = 255
trainimgs = np.asfarray(traindata[:, 1:]) / factor
testimgs = np.asfarray(testdata[:, 0:]) / factor
trainlabels = np.asfarray(traindata[:, :1])

lr = np.arange(10)



onehot = (lr==trainlabels).astype(np.float)

onehot[onehot==0] = 0.01
onehot[onehot==1] = 0.99


import random
np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.e ** -x)

def dsigmoid(x):
    return x*(1-x)

class NN:
    
    def __init__(self, innodes, outnodes, hiddennodes,):
        self.innodes = innodes
        self.outnodes = outnodes
        self.hiddennodes = hiddennodes
        self.weightmatrices()
        
    def weightmatrices(self):
    
    #   initialized weights randomly
       
       
        self.i = [[random.uniform(-1,1) for i in range(self.innodes)] for i in range(self.hiddennodes)]
        #self.i = np.random.random((self.no_of_hidden_nodes, self.no_of_in_nodes))
        
        self.o = [[random.uniform(-1,1) for i in range(self.hiddennodes)] for i in range(self.outnodes)]
       
        #self.o = np.random.random((self.no_of_out_nodes, self.no_of_hidden_nodes))
        
   
        
class trainnet(NN):
    def __init__(self,innodes,outnodes,hiddennodes,
                 learningrate):
        
        super().__init__(innodes,outnodes,hiddennodes)
        self.learningrate=learningrate
    
        self.outer=0
    def train(self, inputV, targetV):
     
        inputV = np.array(inputV, ndmin=2).T
        targetV = np.array(targetV, ndmin=2).T
       
        
        output1 = np.dot(self.i, 
                                inputV)
        outputh = sigmoid(output1)
        
        output2 = np.dot(self.o, 
                                outputh)
        outputn = sigmoid(output2)
        
        outputers = targetV - outputn
 
        self.backprop(outputers,outputn,outputh,inputV)
        error=np.sum((outputers*outputers))/10
        self.outer=self.outer+error
    def backprop(self,outputers,outputn,outputh,inputV):
        
        #update
        t = outputers *dsigmoid(outputn)     
        t = self.learningrate  * np.dot(t, 
                                           outputh.T)
        self.o += t
        if epoch==iter:
            
            print("hidden to output layers weights:",self.o)
        
        hiddeners = np.dot(self.o.T, 
                               outputers)
        
        t = hiddeners * dsigmoid(outputh)
        self.i += self.learningrate * np.dot(t, inputV.T)
        if epoch==iter:
            print("input to hidden layer weights:",self.i)
    
    def runo(self, inputV):
        inputV = np.array(inputV,ndmin=2).T
      
        outputV = np.dot(self.i, 
                               inputV)
        outputV = sigmoid(outputV)
        
        outputV = np.dot(self.o, 
                               outputV)
        outputV = sigmoid(outputV)
        
        return outputV
            
    
        
 
    def checkT(self, data):
        
        with open('/home/ajit/Desktop/finalsub5.csv', 'a') as newFile:
            
            newFileWriter = csv.writer(newFile)
            
            for i in range(len(data)):
                res = self.runo(data[i])
                resmax = res.argmax()
                newFileWriter.writerow([i+1, resmax])
                #writer.writerows(res_max)
                print(resmax)
tr=trainnet(innodes = 784, outnodes = 10, hiddennodes=hidnodes,learningrate = learningrate)
eps =50


for epoch in range(eps):  
    print("ep_nu", epoch)
    for i in range(len(trainimgs)): 
        tr.train(trainimgs[i], onehot[i])
    print(tr.outer)
    tr.outer=0
    
tr.checkT(testimgs)

    







# In[ ]:





# In[ ]:




# In[44]:


import csv

import numpy as np


datapath = "/home/ajit/Downloads/ell409demo1/"
# /home/ajit/Downloads/ell409demo1
# train.csv
traindata = np.loadtxt(datapath + "train.csv", 
                        delimiter=",")
testdata = np.loadtxt(datapath + "UL_Test.csv", 
                       delimiter=",") 


# x=np.asarray(traindata[1])
# print(x)
# mx=max(traindata[1])
# x/mx

for i in range(len(traindata)):
    m=max(traindata[i,1:])
    traindata[i,1:]=traindata[i,1:]


for j in range(len(testdata)):
    m=max(testdata[j,1:])
    testdata[j]
#     x=np.asarray(testdata[j,1:])
    testdata[j,1:]=testdata[j,1:]

    
trainimgs=np.asfarray(traindata[:,1:])
testimgs=np.asfarray(testdata[:,0:])
trainlabels = np.asfarray(traindata[:, :1])
lr = np.arange(12)
onehot = (lr==trainlabels).astype(np.float)
print(trainlabels)
import random
np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.e ** -x)

def dsigmoid(x):
    return x*(1-x)

class NN:
    
    def __init__(self, innodes, outnodes, hiddennodes,):
        self.innodes = innodes
        self.outnodes = outnodes
        self.hiddennodes = hiddennodes
        self.weightmatrices()
        
    def weightmatrices(self):
    
    #   initialized weights randomly
       
       
        self.i = [[random.uniform(-1,1) for i in range(self.innodes)] for i in range(self.hiddennodes)]
        #self.i = np.random.random((self.no_of_hidden_nodes, self.no_of_in_nodes))
        
        self.o = [[random.uniform(-1,1) for i in range(self.hiddennodes)] for i in range(self.outnodes)]
       
        #self.o = np.random.random((self.no_of_out_nodes, self.no_of_hidden_nodes))
        
   
        
class trainnet(NN):
    def __init__(self,innodes,outnodes,hiddennodes,
                 learningrate):
        
        super().__init__(innodes,outnodes,hiddennodes)
        self.learningrate=learningrate
    
        self.outer=0
    def train(self, inputV, targetV):
     
        inputV = np.array(inputV, ndmin=2).T
        targetV = np.array(targetV, ndmin=2).T
       
        
        output1 = np.dot(self.i, 
                                inputV)
        outputh = sigmoid(output1)
        
        output2 = np.dot(self.o, 
                                outputh)
        outputn = sigmoid(output2)
        
        outputers = targetV - outputn
        print(outputers)
        self.backprop(outputers,outputn,outputh,inputV)
        error=np.sum((outputers*outputers))/10
        self.outer=self.outer+error
    def backprop(self,outputers,outputn,outputh,inputV):
        
        #update
        t = outputers *dsigmoid(outputn)     
        t = self.learningrate  * np.dot(t, 
                                           outputh.T)
        self.o += t
        if epoch==iter:
            
            print("hidden to output layers weights:",self.o)
        
        hiddeners = np.dot(self.o.T, 
                               outputers)
        
        t = hiddeners * dsigmoid(outputh)
        self.i += self.learningrate * np.dot(t, inputV.T)
        if epoch==iter:
            print("input to hidden layer weights:",self.i)
    
    def runo(self, inputV):
        inputV = np.array(inputV,ndmin=2).T
      
        outputV = np.dot(self.i, 
                               inputV)
        outputV = sigmoid(outputV)
        
        outputV = np.dot(self.o, 
                               outputV)
        outputV = sigmoid(outputV)
        
        return outputV
            
    
        
 
    def checkT(self, data):
        
        with open('/home/ajit/Desktop/Demo9.csv', 'a') as newFile:
            
            newFileWriter = csv.writer(newFile)
            
            for i in range(len(data)):
                res = self.runo(data[i])
                resmax = res.argmax()
                newFileWriter.writerow([i+1, resmax])
                #writer.writerows(res_max)
                print(resmax)
tr=trainnet(innodes = 10, outnodes = 12, hiddennodes=6,learningrate = 0.01)
eps =2
for epoch in range(eps):  
#     print("ep_nu", epoch)
    for i in range(len(trainimgs)): 
        tr.train(trainimgs[i], onehot[i])
#     print(tr.outer)
    tr.outer=0
    
tr.checkT(testimgs)


# In[24]:


data=[[]]


# In[ ]:




