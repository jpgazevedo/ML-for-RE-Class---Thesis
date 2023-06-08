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

#This script will use the selected AL strategy to select the next training batch to be labeled

#Get unique labels
def get_unique_train_lbls():
    train_lbls = []
    uniqueLbls = []
    with open('Python Scripts/Files/TrainSet.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                aux = line.split(':')
                train_lbls.append(aux[0].lower())
                for i in train_lbls:
                    if not (i in uniqueLbls):
                        uniqueLbls.append(i)

    return np.array(uniqueLbls)

#Main method of AL
def do_strategy(train_set_rest, train_rest_pred, train_rest_probpred, al_strategy, b_size, ml_flg):
    if(al_strategy=='LC'):
        newReq, train_set_rest  = least_confident(train_set_rest, train_rest_pred, train_rest_probpred, b_size, ml_flg)
    if(al_strategy=='MS'):
        newReq, train_set_rest  = margin_sampling(train_set_rest, train_rest_probpred, b_size)
    if(al_strategy=='EM'):
        newReq, train_set_rest  = entropy_measure(train_set_rest, train_rest_probpred, b_size)
    
    return newReq, train_set_rest 

#Least Confident AL strategy
def least_confident(train_set_rest, train_rest_pred, train_rest_probpred, b_size, ml_flg):
    if ml_flg=='NN':
        nlabels=['1','2','3','4','5','6','7','8']
    else:
        nlabels = get_unique_train_lbls()
    probability_prediction = []
    
    for i in range(len(train_rest_probpred)):
         for j in range(len(nlabels)):
             if(train_rest_pred[i]==nlabels[j]):
                 probability_prediction.append((train_rest_probpred[i][j]))
                          
    indexes = train_set_rest.index
    indexed_probs = pd.Series(list(probability_prediction), index=indexes)
    dic = {"requirements": train_set_rest,"probabilities": indexed_probs} 
    data_frame = pd.concat(dic, axis = 1) 
    sorted_data = data_frame.sort_values(by=['probabilities'])
    train_set_rest = sorted_data["requirements"][b_size:]
    req_to_add = sorted_data["requirements"][:b_size]
    
    return req_to_add , train_set_rest

#Margin Sampling AL strategy 
def margin_sampling(train_set_rest, train_rest_probpred, b_size):
    dif = []
    for i in range(len(train_rest_probpred)):
        dictionary = {"probabilities": pd.Series(list(train_rest_probpred[i]))} 
        data_frame = pd.concat(dictionary, axis = 1) 
        sortedMostProb = data_frame.sort_values(by=['probabilities'], ascending=False)
        sortedP = sortedMostProb['probabilities']
        dif.append(sortedP.iloc[0] - sortedP.iloc[1])
        
    indexes = train_set_rest.index
    
    indexed_dif = pd.Series(list(dif), index=indexes)
    dic = {"requirements": train_set_rest,"differences": indexed_dif} 
    data_frame = pd.concat(dic, axis = 1) 
    sorted_data = data_frame.sort_values(by=['differences'])
    train_set_rest = sorted_data["requirements"][b_size:]
    req_to_add = sorted_data["requirements"][:b_size]

    return req_to_add , train_set_rest


#Entropy Mesure AL strategy
def entropy_measure(train_set_rest, train_rest_probpred, b_size):
    entropies = []
    for i in range(len(train_rest_probpred)):
        aux = pd.Series(train_rest_probpred[i])
        counts = aux.value_counts()
        v = entropy(counts)
        entropies.append(v)
        
    indexes = train_set_rest.index
        
    indexed_ent = pd.Series(list(entropies), index=indexes)
    dic = {"requirements": train_set_rest,"entropy": indexed_ent} 
    data_frame = pd.concat(dic, axis = 1) 
    sorted_data = data_frame.sort_values(by=['entropy'])
    train_set_rest = sorted_data["requirements"][b_size:]
    req_to_add = sorted_data["requirements"][:b_size]

    return req_to_add , train_set_rest

#Remove accents
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

#Extra step for user stories. Remove roles
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

#Run MNB model   
def run_MNB_AL(rest_train_data, al_flg):
    train_p = rest_train_data['processed']

    with open('Python Scripts/Models/NBModel.pkl', 'rb') as f:
        countvec, model = pickle.load(f)


    train_fts = countvec.transform(train_p)

    lbls_pred = model.predict(train_fts)
    prob_pred = model.predict_proba(train_fts)

    return do_strategy(rest_train_data['unprocessed'], lbls_pred, prob_pred, al_flg, 20, 'MNB')

#Run SVC model
def run_SVC_AL(rest_train_data, al_flg):
    train_p = rest_train_data['processed']

    with open('Python Scripts/Models/SVCModel.pkl', 'rb') as f:
        countvec, model = pickle.load(f)

    train_fts = countvec.transform(train_p)
    
    lbls_pred = model.predict(train_fts)
    prob_pred = model.predict_proba(train_fts)

    return do_strategy(rest_train_data['unprocessed'], lbls_pred, prob_pred, al_flg, 20, 'SVC')

#Run LR model
def run_LR_AL(rest_train_data, al_flg):
    train_p = rest_train_data['processed']

    with open('Python Scripts/Models/LRModel.pkl', 'rb') as f:
        countvec, model = pickle.load(f)

    train_fts = countvec.transform(train_p)
    
    lbls_pred = model.predict(train_fts)
    prob_pred = model.predict_proba(train_fts)

    return do_strategy(rest_train_data['unprocessed'], lbls_pred, prob_pred, al_flg, 20, 'LR')


#Get the predicted lbl with the predicted probabilities
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
 
    return pre_pred

#Run NN model
def run_NN_AL(rest_train_data, al_flg):
    train_p = rest_train_data['processed']

    model=tf.keras.models.load_model('Python Scripts/Models/NNModel')

    prob_pred = model.predict(train_p, verbose=0)

    lbls_pred = get_yPred(prob_pred)

    return do_strategy(rest_train_data['unprocessed'], lbls_pred, prob_pred, al_flg, 20, 'NN')

#Main method
def run(ml_flg, al_flg, re_flg):
    unp_train_rest = []
    p_train_rest = []
    with open('Python Scripts/Files/RestOfTrainSet.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            unp_train_rest.append(line.replace(';','').replace('\n',''))
        unp_train_rest=np.array(unp_train_rest)

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

    train_rest = {'processed': p_train_rest, 'unprocessed':unp_train_rest}

    train_rest_df = pd.DataFrame(train_rest)

    if (ml_flg=='MNB'):
        nextTrainBatch, restTrain = run_MNB_AL(train_rest_df, al_flg)
    elif (ml_flg=='SVC'):
        nextTrainBatch, restTrain = run_SVC_AL(train_rest_df, al_flg)
    elif (ml_flg=='LR'):
        nextTrainBatch, restTrain = run_LR_AL(train_rest_df, al_flg)
    else:
        nextTrainBatch, restTrain = run_NN_AL(train_rest_df, al_flg)

    with open('Python Scripts/Files/RestOfTrainSet.txt', 'w') as file:
            i = 0
            for req in restTrain:
                if i == len(restTrain)-1:
                    file.write(req+';')
                else:
                    file.write(req+';'+'\n')
                i=i+1
    
    with open('Python Scripts/Files/NextTrainBatch.txt', 'w') as file:
            i = 0
            for req in nextTrainBatch:
                if i == len(nextTrainBatch)-1:
                    file.write(req+';')
                else:
                    file.write(req+';'+'\n')
                i=i+1

    out = ''
    for nextReq in nextTrainBatch:
        out = out + nextReq.replace('\n', '').replace('\'', '')+';'

    return out
    
input = sys.argv[1]

input = input.split(',')

trainBatchData = run(input[0], input[1], input[2])

print(trainBatchData)
sys.stdout.flush()