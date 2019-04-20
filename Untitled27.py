#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


print("Enter No of hidden nodes")
hidnodes=int(input())
print("ENter learning rate")
learning_rate=float(input())
import csv
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "/home/ajit/Desktop/ML/ell-409-assignment-1/"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 

fac = 255
train_imgs = np.asfarray(train_data[:, 1:]) / fac
test_imgs = np.asfarray(test_data[:, 0:]) / fac
train_labels = np.asfarray(train_data[:, :1])
#test_labels = np.asfarray(test_data[:, :1])
lr = np.arange(no_of_different_labels)
# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
#test_labels_one_hot = (lr==test_labels).astype(np.float)
# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
#test_labels_one_hot[test_labels_one_hot==0] = 0.01
#test_labels_one_hot[test_labels_one_hot==1] = 0.99
import numpy as np
import random
np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid
def dsigmoid(x):
    return x*(1-x)

class NeuralNetwork:
    
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        """ 
       initialized weights randomly
        """
       
        self.i = [[random.uniform(-1,1) for i in range(self.no_of_in_nodes)] for i in range(self.no_of_hidden_nodes)]
        #self.i = np.random.random((self.no_of_hidden_nodes, self.no_of_in_nodes))
        
        self.o = [[random.uniform(-1,1) for i in range(self.no_of_hidden_nodes)] for i in range(self.no_of_out_nodes)]
       
        #self.o = np.random.random((self.no_of_out_nodes, self.no_of_hidden_nodes))
        
   
        
class train_net(NeuralNetwork):
    def __init__(self,no_of_in_nodes,no_of_out_nodes,no_of_hidden_nodes,
                 learning_rate):
        
        super().__init__(no_of_in_nodes,no_of_out_nodes,no_of_hidden_nodes)
        self.learning_rate=learning_rate
    
        self.outer=0
    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can 
        be tuple, list or ndarray
        """
        
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
       
        
        output_vector1 = np.dot(self.i, 
                                input_vector)
        output_hidden = activation_function(output_vector1)
        
        output_vector2 = np.dot(self.o, 
                                output_hidden)
        output_network = activation_function(output_vector2)
        
        output_errors = target_vector - output_network
        
        error=np.sum((output_errors*output_errors))/10
        self.outer=self.outer+error
       
        # update the weights:
        tmp = output_errors *dsigmoid(output_network)     
        tmp = self.learning_rate  * np.dot(tmp, 
                                           output_hidden.T)
        self.o += tmp
        if epoch==iter:
            
            print("hidden to output layers weights:",self.o)
        # calculate hidden errors:
        hidden_errors = np.dot(self.o.T, 
                               output_errors)
        # update the weights:
        tmp = hidden_errors * dsigmoid(output_hidden)
        self.i += self.learning_rate                           * np.dot(tmp, input_vector.T)
        if epoch==iter:
            print("input to hidden layer weights:",self.i)
    
    def run(self, input_vector):
        input_vector = np.array(input_vector,ndmin=2).T
      
        output_vector = np.dot(self.i, 
                               input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.o, 
                               output_vector)
        output_vector = activation_function(output_vector)
        
        return output_vector
            
    
        
    
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            if epoch==iter:
                print("output_vector_of_the_input:",res)
                print("one_hot_representataion:",(lr==labels[i]).astype(np.float))
            res_max = res.argmax()
                
           # print(res_max,labels[i])
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs
    def evaluateT(self, data):
        corrects, wrongs = 0, 0
        with open('/home/ajit/Desktop/finalsub4.csv', 'a') as newFile:
            
            newFileWriter = csv.writer(newFile)
            
            for i in range(len(data)):
                res = self.run(data[i])
                res_max = res.argmax()
                newFileWriter.writerow([i+1, res_max])
                #writer.writerows(res_max)
                print(res_max)
tr=train_net(no_of_in_nodes = image_pixels, 
                   no_of_out_nodes = 10, 
                   no_of_hidden_nodes=hidnodes,
                   learning_rate = learning_rate)
epochs = 500


for epoch in range(epochs):  
    print("epoch: ", epoch)
    for i in range(len(train_imgs)):
        
        tr.train(train_imgs[i], 
                 train_labels_one_hot[i])
    print(tr.outer)
    tr.outer=0
    
tr.evaluateT(test_imgs)

    


# In[ ]:




