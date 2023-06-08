import pandas as pd
import numpy as np
from scipy.stats import entropy
import sys
import re
import unicodedata
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import tensorflow_text as text

#Remove accents
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

#Extra preprocessing step for user stories. Remove roles
def removeRole(req):
    if not(',' in req):
        return req
    else:
        aux = req.split(',')
        i=1
        finalUS = ''
        while i < len(aux):
            if i==1:
                finalUS = finalUS + aux[i]
            elif i==len(aux)-1:
                finalUS = finalUS + aux[i]
            else:
                finalUS = finalUS + aux[i]
            i=i+1
        return finalUS

#Transform the predicted prob in real lbls
def get_yPred(yPred):
    pre_pred = []
    i =0
    aux1 = []
    while(i<len(yPred)):
        j=0
        aux1 = yPred[i]
        aux=0
        fValue=0
        while(j<len(aux1)):
            aux2 = str(aux1[j])
            aux2ARR = aux2.split(".")
            aux2STR = aux2ARR[0]+aux2ARR[1]
            auxINT = int(aux2STR[:4])
            if(auxINT > aux):
                aux=auxINT
                fValue=str(j+1)
            j=j+1
        i=i+1
        pre_pred.append(fValue)
    pre_pred = np.array(pre_pred)

    #Read labels map
    lbls_map = []
    with open('Python Scripts/Files/NN_LBLS_MAP.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            aux = line.split(';')
            lbls_map.append((aux[0],aux[1]))
        lbls_map=np.array(lbls_map)

    strLbl = lbls_map[:,0]
    intLbl = lbls_map[:,1]
    i=0
    result = []
    while i < len(intLbl):
        result = [sub.replace(intLbl[i], strLbl[i]) for sub in pre_pred]
        i=i+1

    return result

#Run MNB model
def run_MNB(rest_train_data):
    train_p = rest_train_data['processed']
    train_up = rest_train_data['unprocessed']

    with open('Python Scripts/Models/NBModel.pkl', 'rb') as f:
        countvec, model = pickle.load(f)

    train_fts = countvec.transform(train_p)

    lbls_pred = model.predict(train_fts)
    
    rest_train_data_final = []
    i = 0
    for lbl in lbls_pred:
        rest_train_data_final.append((lbl.upper(), train_up[i]))
        i=i+1
    return np.array(rest_train_data_final)

#Run SVC model
def run_SVC(rest_train_data):
    train_p = rest_train_data['processed']
    train_up = rest_train_data['unprocessed']

    with open('Python Scripts/Models/SVCModel.pkl', 'rb') as f:
        countvec, model = pickle.load(f)

    train_fts = countvec.transform(train_p)

    lbls_pred = model.predict(train_fts)
    
    rest_train_data_final = []
    i = 0
    for lbl in lbls_pred:
        rest_train_data_final.append((lbl.upper(), train_up[i]))
        i=i+1
    return np.array(rest_train_data_final)
    

#Run LR model
def run_LR(rest_train_data):
    train_p = rest_train_data['processed']
    train_up = rest_train_data['unprocessed']

    with open('Python Scripts/Models/LRModel.pkl', 'rb') as f:
        countvec, model = pickle.load(f)


    train_fts = countvec.transform(train_p)

    lbls_pred = model.predict(train_fts)
    
    rest_train_data_final = []
    i = 0
    for lbl in lbls_pred:
        rest_train_data_final.append((lbl.upper(), train_up[i]))
        i=i+1
    return np.array(rest_train_data_final)

#Run NN model
def run_NN(rest_train_data):
    train_p = rest_train_data['processed']
    train_up = rest_train_data['unprocessed']

    model=tf.keras.models.load_model('Python Scripts/Models/NNModel')

    prob_pred = model.predict(train_p, verbose=0)

    lbls_pred = get_yPred(prob_pred)

    rest_train_data_final = []
    i = 0
    for lbl in lbls_pred:
        rest_train_data_final.append((lbl.upper(), train_up[i]))
        i=i+1

    return np.array(rest_train_data_final)

# Main method
def run(ml_flg, re_flg):
    unp_train_rest = []
    with open('Python Scripts/Files/RestOfTrainSet.txt', 'r') as file:
        lines = file.readlines()
        restOfTrainSize = len(lines)
        if restOfTrainSize>0:
            for line in lines:
                unp_train_rest.append(line.replace(';','').replace('\n',''))
            unp_train_rest=np.array(unp_train_rest)

    train_set = []
    with open('Python Scripts/Files/TrainSet.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            aux = line.split(':')
            train_set.append((aux[0], aux[1].replace(';','').replace('\n','')))
        train_set=np.array(train_set)

    test_set = []
    with open('Python Scripts/Files/TestSet.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            aux = line.split(':')
            test_set.append((aux[0], aux[1].replace(';','').replace('\n','')))
        test_set=np.array(test_set)

    final_set = np.append(train_set, test_set)
    if restOfTrainSize>0:
        p_train_rest = []
        for req in unp_train_rest:
            noWeirdChars = re.sub(r"[^a-zA-Z]+", ' ', strip_accents(req).lower().strip())
            A = re.sub(r'\b\w{1,3}\b', '', noWeirdChars)
            B = re.sub(" shall "," ",A)
            C = re.sub(" as "," ",B)
            D = re.sub(" that "," ",C)
            noWeirdChars = re.sub(r"[^a-zA-Z]+", ' ', D)
            noLeadingWhiteSpace = noWeirdChars.lstrip()
            noTrailingWhiteSpace = noLeadingWhiteSpace.rstrip()
            if (re_flg=='US'):
                noTrailingWhiteSpace = removeRole(noTrailingWhiteSpace)
            p_train_rest.append(noTrailingWhiteSpace)    
        p_train_rest = np.array(p_train_rest)

        train_rest = {'processed': p_train_rest, 'unprocessed': unp_train_rest}
        train_rest_df = pd.DataFrame(train_rest)

        rest_labeled = []
        if (ml_flg=='MNB'):
            rest_labeled = run_MNB(train_rest_df)
        elif (ml_flg=='SVC'):
            rest_labeled = run_SVC(train_rest_df)
        elif (ml_flg=='LR'):
            rest_labeled = run_LR(train_rest_df)
        else:
            rest_labeled = run_NN(train_rest_df)

        aux_app = np.append(test_set, train_set, axis=0)
        final_set = np.append(aux_app, rest_labeled, axis=0)
    else:
        final_set = np.append(test_set, train_set, axis=0)   

    with open('src/files/Classified_Dataset.txt', 'w') as file:
            i = 0
            lbls = final_set[:,0]
            reqs = final_set[:,1]
            for req in reqs:
                if i == len(reqs)-1:
                    if lbls[i] == 'F':
                        file.write('Functional'+':'+str(req)+';')
                    elif lbls[i] == 'US':
                        file.write('Usability'+':'+str(req)+';')
                    elif lbls[i] == 'MN':
                        file.write('Maintainability'+':'+str(req)+';')
                    elif lbls[i] == 'PE':
                        file.write('Performance'+':'+str(req)+';')
                    elif lbls[i] == 'CO':
                        file.write('Compatability'+':'+str(req)+';')
                    elif lbls[i] == 'RE':
                        file.write('Reliability'+':'+str(req)+';')
                    elif lbls[i] == 'PO':
                        file.write('Portability'+':'+str(req)+';')
                    elif lbls[i] == 'SE':
                        file.write('Secutiry'+':'+str(req)+';')
                else:
                    if lbls[i] == 'F':
                        file.write('Functional'+':'+str(req)+';'+'\n')
                    elif lbls[i] == 'US':
                        file.write('Usability'+':'+str(req)+';'+'\n')
                    elif lbls[i] == 'MN':
                        file.write('Maintainability'+':'+str(req)+';'+'\n')
                    elif lbls[i] == 'PE':
                        file.write('Performance'+':'+str(req)+';'+'\n')
                    elif lbls[i] == 'CO':
                        file.write('Compatability'+':'+str(req)+';'+'\n')
                    elif lbls[i] == 'RE':
                        file.write('Reliability'+':'+str(req)+';'+'\n')
                    elif lbls[i] == 'PO':
                        file.write('Portability'+':'+str(req)+';'+'\n')
                    elif lbls[i] == 'SE':
                        file.write('Secutiry'+':'+str(req)+';'+'\n')
                i=i+1
    

input = sys.argv[1]

input = input.split(',')
run(input[0], input[1])
print('')
sys.stdout.flush()