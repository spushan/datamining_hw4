#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

per_high_acc, per_best_n, per_best_r, per_best_a = 0, 0, True, 0
mlp_high_acc, mlp_best_n, mlp_best_r, mlp_best_a = 0, 0, True, 0
for w in n: #iterates over n
    for b in r: #iterates over r
        for a in range(2): #iterates over the algorithms

            #Create a Neural Network classifier
            if a==0:
               clf = Perceptron(eta0=w, shuffle=b, max_iter=1000)
            else:
               clf = MLPClassifier(activation='logistic', learning_rate_init=w, hidden_layer_sizes=(25,), shuffle =b, max_iter=1000)     
            clf.fit(X_training, y_training)

            if a==0:
                corr=0
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    pred = clf.predict([x_testSample])
                    if pred[0] == y_testSample:
                        corr += 1 
                acc = round(corr / len(X_test) * 100, 2)
                if acc > per_high_acc:
                    per_high_acc = acc
                    per_best_n = w
                    per_best_r = b
                    print(f"Highest Perceptron accuracy so far: {acc}, Parameters: learning rate= {per_best_n}, shuffle= {per_best_r}")
            else:
                corr=0
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    pred = clf.predict([x_testSample])
                    if pred[0] == y_testSample:
                        corr += 1 
                acc = round(corr / len(X_test) * 100, 2)
                if acc > mlp_high_acc:
                    mlp_high_acc = acc
                    mlp_best_n = w
                    mlp_best_r = b      
                    print(f"Highest MLP accuracy so far: {acc}, Parameters: learning rate= {mlp_best_n}, shuffle= {mlp_best_r}")
               
      











