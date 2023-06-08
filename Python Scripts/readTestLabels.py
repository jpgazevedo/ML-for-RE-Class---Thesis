import pandas as pd;
import numpy as np;
from pandas.plotting import table 
from sklearn.model_selection import train_test_split
import sys

#This script will read the labeled test set and write the file

def run(input):

    aux_arr = input.split(',')

    #Get Test set to label
    reqs = []
    with open('Python Scripts/Files/TestSet_PreLbl.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            
            aux = line.split(';')
            for i in range(0,len(aux)):
                if (aux[i]!='' and aux[i]!='\n' and aux[i]!=' '):
                    reqs.append(aux[i]+';')
    np.array(reqs)

    raw=[]
    i=0
    for req in reqs:
        aux = []
        raw.append((req, aux_arr[i]))
        i+=1
                         


    #Write Test set with labels
    with open('Python Scripts/Files/TestSet.txt', 'w') as f:
        j=0
        for line in raw:
            if j == len(raw)-1:
                f.write(line[1] + ': ' + line[0].replace('\n', ''))
            else:
                f.write(line[1] + ': ' + line[0].replace('\n', '')+'\n')
            j=j+1
        
    #Retrieve the first batch of train set
    out = ''
    with open('Python Scripts/Files/NextTrainBatch.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            aux = line.split(';')
            for i in range(0, len(aux)):
                if (aux[i]!='' and aux[i]!='\n' and aux[i]!=' '):
                    out = out + aux[i].replace('\n', '').replace('\'', '')+';'
    return out

input = sys.argv[1]

trainBatchData = run(input)

print(trainBatchData)
sys.stdout.flush()