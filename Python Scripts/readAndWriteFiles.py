import pandas as pd;
import numpy as np;
from pandas.plotting import table 
import matplotlib.pyplot as plt
import dataframe_image as dfi
import os
from sklearn.model_selection import train_test_split
import sys
import json
import ast
import time

#This script will read the input submited text file. And write the sets

def run():
    data = []
    with open('Python Scripts/Files/initialData.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            
            aux = line.split(';')
            for i in range(0,len(aux)):
                if (aux[i]!='' and aux[i]!='\n' and aux[i]!=' '):
                    data.append(aux[i]+';')

        
    np.array(data)
    np.random.shuffle(data)
    dataTrain, dataTest = train_test_split(data, test_size=0.200, shuffle=True)
    np.random.shuffle(dataTrain)

    batchTrain = []
    for i in range(0,20):
        batchTrain.append(dataTrain[i])
    
    dataTrain = np.delete(dataTrain, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])


    with open('Python Scripts/Files/RestOfTrainSet.txt', 'w') as f:
        for i in range(0,len(dataTrain)):
            if i == len(dataTrain)-1:
                f.write(dataTrain[i])
            else:
                f.write(dataTrain[i]+'\n')

    with open('Python Scripts/Files/NextTrainBatch.txt', 'w') as f:
        for i in range(0,len(batchTrain)):
            if i == len(batchTrain)-1:
                f.write(batchTrain[i])
            else:
                f.write(batchTrain[i]+'\n')

    with open('Python Scripts/Files/TestSet_PreLbl.txt', 'w') as tf:
        for i in range(0,len(dataTest)):
            if i == len(dataTest)-1:
                tf.write(dataTest[i])
            else:
                tf.write(dataTest[i]+'\n')

    
    return dataTest

def prepareOutput(data):
    str =''
    for i in range(0, len(data)):
        str = str+data[i]
    return str




testData = run()

preparedOut = prepareOutput(testData)

print(preparedOut)
sys.stdout.flush()