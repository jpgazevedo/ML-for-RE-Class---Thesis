from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, mean_squared_log_error, precision_score, f1_score, cohen_kappa_score
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import unicodedata
import re
import sys
import tensorflow_text as text

#This script will train the selected classifier with the actual training set

#Transform probs predicted in real labels
def get_yPred(yPred):
    result = []
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
                fValue=j+1
            j=j+1
        i=i+1
        result.append(fValue)
    return np.array(result)

#Remove accents
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

#Get already used percentage
def get_usedperc(totalUsed):
    n = 0
    with open('Python Scripts/Files/RestOfTrainSet.txt', 'r') as file:
        lines = file.readlines()
        n=len(lines)
    n = n + totalUsed

    return totalUsed*100/n
    
#Returns the test set requirements and labels
def getTestData(re_type):
    raw = []
    processedData = []
    aux = []
    
    with open('Python Scripts/Files/TestSet.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            aux = line.split(':')
            aux[1] = strip_accents(aux[1]).lower().strip()
            aux[0] = strip_accents(aux[0]).lower().strip()
            raw.append((aux[0], aux[1]))
                        
        raw = np.array(raw)
                  
        for req in raw:
            noWeirdChars = re.sub(r"[^a-zA-Z]+", ' ', strip_accents(req[1]).lower().strip())
            A = re.sub(r'\b\w{1,3}\b', '', noWeirdChars)
            B = re.sub(" shall "," ",A)
            C = re.sub(" as "," ",B)
            D = re.sub(" that "," ",C)
            noWeirdChars = re.sub(r"[^a-zA-Z]+", ' ', D)
            noLeadingWhiteSpace = noWeirdChars.lstrip()
            noTrailingWhiteSpace = noLeadingWhiteSpace.rstrip()
            if (re_type=='US'):
                noTrailingWhiteSpace = removeRole(noTrailingWhiteSpace)
            processedData.append((strip_accents(req[0]).lower().strip(), noTrailingWhiteSpace))    
        processedData = np.array(processedData)

    return  processedData[:,0], processedData[:,1]

#Calculate the recall
def getRecall(pre, f1):
  return round(((pre*f1)/(2*pre - f1)),1)

##Build the neural network
def buildNNModel():
    tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/2"
    tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['sequence_output']
    net = tf.keras.layers.SpatialDropout1D(0.5)(net)
    net = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(net)
    net = tf.keras.layers.Dropout(0.8)(net)
    net = tf.keras.layers.Dense(64, activation='relu', name='Dense1')(net)
    net = tf.keras.layers.Dropout(0.8)(net)
    net = tf.keras.layers.Dense(32, activation='relu', name='Dense2')(net)
    net = tf.keras.layers.Dropout(0.8)(net)
    net = tf.keras.layers.Dense(8, activation='softmax', name='Dense3')(net)

    model = tf.keras.Model(text_input, net)
    
    return model

#Transform the labels into int      
def transform_lbls(predicted_lbls):
    intDif = []
    strDif = []
    result = []
    i = 0
    for lbl in predicted_lbls:
        if not (lbl in strDif):
            strDif.append(lbl)
            intDif.append(i)
            i=i+1
    for lbl in predicted_lbls:
        j = 0
        for strD in strDif:
            if (strD==lbl):
                result.append(intDif[j])
            j = j+1

    return result

#Run the MNB model
def run_MNB(firstItFlag, typeOfRE, trainData, restOfTrain):
    testLbls, testReq = getTestData(typeOfRE)
    trainData = np.array(trainData)
    trainLbls = trainData[:,0]
    trainReq = trainData[:,1]

    totalREQS = np.append(testReq,trainReq)
    totalREQS = np.append(totalREQS,restOfTrain)
    
    if (firstItFlag=='Y'):
        countvec = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
        countvec.fit(totalREQS)
        if typeOfRE=='RE':
                model = MultinomialNB(alpha=0.3, fit_prior=True)
        elif typeOfRE=='US':
                model = MultinomialNB(alpha=0.3, fit_prior=True)
    else:
        with open('Python Scripts/Models/NBModel.pkl', 'rb') as f:
            countvec, model = pickle.load(f)

    
    train_fts = countvec.transform(trainReq)
    model.fit(train_fts, trainLbls)
    model.fit(train_fts, trainLbls)

    test_fts = countvec.transform(testReq)
    predictions = model.predict(test_fts)

    intPred = transform_lbls(predictions)
    intTest = transform_lbls(testLbls)

    accuracy=accuracy_score(testLbls, predictions)
    msle=mean_squared_log_error(intTest, intPred)
    precision=precision_score(testLbls, predictions, average='weighted', zero_division=1)
    f1=f1_score(testLbls, predictions, average='weighted')
    recall=getRecall(precision,f1)

    if (firstItFlag=='Y'):
        kappa=0.0
    else:
        kappa=cohen_kappa_score(intTest, intPred)

    with open('Python Scripts/Models/NBModel.pkl', 'wb') as fout:
        pickle.dump((countvec, model), fout)

    perc = get_usedperc(len(trainReq)+len(testReq))

    return accuracy, msle, precision, recall, f1, kappa, perc

#Run the SVC model
def run_SVC(firstItFlag, typeOfRE, trainData, restOfTrain):
    testLbls, testReq = getTestData(typeOfRE)
    trainData = np.array(trainData)
    trainLbls = trainData[:,0]
    trainReq = trainData[:,1]

    totalREQS = np.append(testReq,trainReq)
    totalREQS = np.append(totalREQS,restOfTrain)

    if (firstItFlag=='Y'):
        countvec = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
        countvec.fit(totalREQS)
        if (typeOfRE=='RE'):
                model = SVC(kernel='linear', C=1, probability=True)
        elif (typeOfRE=='US'):
                model = SVC(kernel='linear', C=0.5, probability=True)
    else:
        with open('Python Scripts/Models/SVCModel.pkl', 'rb') as f:
            countvec, model = pickle.load(f)

    
    train_fts = countvec.transform(trainReq)
    model.fit(train_fts, trainLbls)
    model.fit(train_fts, trainLbls)

    test_fts = countvec.transform(testReq)
    predictions = model.predict(test_fts)

    intPred = transform_lbls(predictions)
    intTest = transform_lbls(testLbls)

    accuracy=accuracy_score(testLbls, predictions)
    msle=mean_squared_log_error(intTest, intPred)
    precision=precision_score(testLbls, predictions, average='weighted', zero_division=1)
    f1=f1_score(testLbls, predictions, average='weighted')
    recall=getRecall(precision,f1)

    if (firstItFlag=='Y'):
        kappa=0.0
    else:
        kappa=cohen_kappa_score(intTest, intPred)

    with open('Python Scripts/Models/SVCModel.pkl', 'wb') as fout:
        pickle.dump((countvec, model), fout)
    
    perc = get_usedperc(len(trainReq)+len(testReq))

    return accuracy, msle, precision, recall, f1, kappa, perc

#Run the LR model
def run_LR(firstItFlag, typeOfRE, trainData, restOfTrain):
    testLbls, testReq = getTestData(typeOfRE)
    trainData = np.array(trainData)
    trainLbls = trainData[:,0]
    trainReq = trainData[:,1]

    totalREQS = np.append(testReq,trainReq)
    totalREQS = np.append(totalREQS,restOfTrain)

    if (firstItFlag=='Y'):
        countvec = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
        countvec.fit(totalREQS)
        if (typeOfRE=='RE'):
                model = LogisticRegression(solver='liblinear', penalty='l2', C=5, fit_intercept=True, class_weight='balanced', max_iter=50000)
        elif (typeOfRE=='US'):
                model = LogisticRegression(solver='liblinear', penalty='l2', C=1000, fit_intercept=False, class_weight=None, max_iter=50000)
    else:
        with open('Python Scripts/Models/LRModel.pkl', 'rb') as f:
            countvec, model = pickle.load(f)


    train_fts = countvec.transform(trainReq)
    model.fit(train_fts, trainLbls)
    model.fit(train_fts, trainLbls)

    test_fts = countvec.transform(testReq)
    predictions = model.predict(test_fts)

    intPred = transform_lbls(predictions)
    intTest = transform_lbls(testLbls)

    accuracy=accuracy_score(testLbls, predictions)
    msle=mean_squared_log_error(intTest, intPred)
    precision=precision_score(testLbls, predictions, average='weighted', zero_division=1)
    f1=f1_score(testLbls, predictions, average='weighted')
    recall=getRecall(precision,f1)

    if (firstItFlag=='Y'):
        kappa=0.0
    else:
        kappa=cohen_kappa_score(intTest, intPred)

    with open('Python Scripts/Models/LRModel.pkl', 'wb') as fout:
        pickle.dump((countvec, model), fout)
    
    perc = get_usedperc(len(trainReq)+len(testReq))

    return accuracy, msle, precision, recall, f1, kappa, perc

#Convert string labels to int and write the mappings
def conv2IntandWriteMap(test_strLbls, train_strLbls):
    strLbls = np.append(test_strLbls, train_strLbls)
    uniqueLbls=strLbls[np.sort(np.unique(strLbls, return_index=True)[1])]
    
    i=0
    with open('Python Scripts/Files/NN_LBLS_MAP.txt', 'w') as file:
        i = 0
        for lbl in uniqueLbls:
            if i == len(uniqueLbls)-1:
                file.write(str(lbl)+';'+str(i+1))
            else:
                file.write(str(lbl)+';'+str(i+1)+'\n')
            i=i+1

    i=0
    while i < len(uniqueLbls):
        train_strLbls = [sub.replace(uniqueLbls[i], str(i+1)) for sub in train_strLbls]
        test_strLbls = [sub.replace(uniqueLbls[i], str(i+1)) for sub in test_strLbls]
        i=i+1

    return np.array(test_strLbls).astype('int'), np.array(train_strLbls).astype('int')

#Run the NN model
def run_NN(firstItFlag, typeOfRE, trainData, restOfTrain):
    testLbls, testReq = getTestData(typeOfRE)
    trainData = np.array(trainData)
    trainLbls = trainData[:,0]
    trainReq = trainData[:,1]

    if (firstItFlag=='Y'):
        model = buildNNModel()
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    else:
        model=tf.keras.models.load_model('Python Scripts/Models/NNModel')
        
    y_test, y_train = conv2IntandWriteMap(testLbls, trainLbls)

    model.fit(trainReq, y_train, epochs=2, verbose=0)

    predicts = model.predict(testReq, verbose=0)

    accuracy=accuracy_score(y_test, get_yPred(predicts))
    msle=mean_squared_log_error(y_test, get_yPred(predicts))
    precision=precision_score(y_test, get_yPred(predicts), average='weighted', zero_division=1)
    recall=getRecall(precision_score(y_test, get_yPred(predicts), average='weighted', zero_division=1), f1_score(y_test, get_yPred(predicts), average='weighted'))
    f1=f1_score(y_test, get_yPred(predicts), average='weighted')

    if (firstItFlag=='Y'):
        kappa = 0.0
    else:
        kappa=cohen_kappa_score(y_test, get_yPred(predicts))

    model.save('Python Scripts/Models/NNModel')

    perc = get_usedperc(len(trainReq)+len(testReq))

    return accuracy, msle, precision, recall, f1, kappa, perc

#Extra preprocessing step for user stories. Remove role
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

#Main method
def run(ml_strategy, re_type, first_flg, input):
     
    raw_data = []

    # Get next train batch to add
    with open('Python Scripts/Files/NextTrainBatch.txt', 'r') as file:
        lines = file.readlines()
        i = 3
        for line in lines:
            raw_data.append((input[i],line.replace('\n', '')))
            i=i+1

        
    processedData = []
    # If first iteration
    if (first_flg=='Y'):
        # Write file with the train set
        with open('Python Scripts/Files/TrainSet.txt', 'w') as file:
            i = 0
            for req in raw_data:
                if i == len(raw_data)-1:
                    file.write(req[0]+': '+ req[1])
                else:
                    file.write(req[0]+': '+ req[1]+'\n')
                i=i+1

        # Process the training data
        for req in raw_data:
            noWeirdChars = re.sub(r"[^a-zA-Z]+", ' ', strip_accents(req[1]).lower().strip())
            A = re.sub(r'\b\w{1,3}\b', '', noWeirdChars)
            B = re.sub(" shall "," ",A)
            C = re.sub(" as "," ",B)
            D = re.sub(" that "," ",C)
            noWeirdChars = re.sub(r"[^a-zA-Z]+", ' ', D)
            noLeadingWhiteSpace = noWeirdChars.lstrip()
            noTrailingWhiteSpace = noLeadingWhiteSpace.rstrip()
            if (re_type=='US'):
                noTrailingWhiteSpace = removeRole(noTrailingWhiteSpace)
            processedData.append((strip_accents(req[0]).lower().strip(), noTrailingWhiteSpace))
            
        np.array(processedData)
    else:
        trainData = []
        # Get training set
        with open('Python Scripts/Files/TrainSet.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                aux = line.split(':')
                trainData.append((aux[0], aux[1].replace('\n', '')))

        
        # Concat training set with next training batch  
        for req in raw_data:
            trainData.append(req)
    
        # Write training set file
        with open('Python Scripts/Files/TrainSet.txt', 'w') as file:
            i = 0
            for req in trainData:
                if (i == len(trainData)-1):
                    file.write(req[0]+': '+ req[1])
                else:
                    file.write(req[0]+': '+ req[1]+'\n')
                i=i+1
                
        
        # Preprocess training set
        for req in trainData:
            noWeirdChars = re.sub(r"[^a-zA-Z]+", ' ', strip_accents(req[1]).lower().strip())
            A = re.sub(r'\b\w{1,3}\b', '', noWeirdChars)
            B = re.sub(" shall "," ",A)
            C = re.sub(" as "," ",B)
            D = re.sub(" that "," ",C)
            noWeirdChars = re.sub(r"[^a-zA-Z]+", ' ', D)
            noLeadingWhiteSpace = noWeirdChars.lstrip()
            noTrailingWhiteSpace = noLeadingWhiteSpace.rstrip()
            if (re_type=='US'):
                noTrailingWhiteSpace = removeRole(noTrailingWhiteSpace)
            processedData.append((strip_accents(req[0]).lower().strip(), noTrailingWhiteSpace))
        processedData = np.array(processedData)

    restOfTrain = []
    with open('Python Scripts/Files/RestOfTrainSet.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                restOfTrain.append(line.replace('\n',''))
    restOfTrain_P = []
    for req in restOfTrain:
            noWeirdChars = re.sub(r"[^a-zA-Z]+", ' ', strip_accents(req).lower().strip())
            A = re.sub(r'\b\w{1,3}\b', '', noWeirdChars)
            B = re.sub(" shall "," ",A)
            C = re.sub(" as "," ",B)
            D = re.sub(" that "," ",C)
            noWeirdChars = re.sub(r"[^a-zA-Z]+", ' ', D)
            noLeadingWhiteSpace = noWeirdChars.lstrip()
            noTrailingWhiteSpace = noLeadingWhiteSpace.rstrip()
            if (re_type=='US'):
                noTrailingWhiteSpace = removeRole(noTrailingWhiteSpace)
            restOfTrain_P.append(noTrailingWhiteSpace)
    
    # Train ML model and obtain acc, msle, pre, rec, f1, k and percentage of used data
    if (ml_strategy=='MNB'):
        acc, msle, pre, rec, f1, k, perc = run_MNB(first_flg, re_type, processedData, restOfTrain_P)
    elif (ml_strategy=='SVC'):
        acc, msle, pre, rec, f1, k, perc = run_SVC(first_flg, re_type, processedData, restOfTrain_P)
    elif (ml_strategy=='LR'):
        acc, msle, pre, rec, f1, k, perc = run_LR(first_flg, re_type, processedData, restOfTrain_P)
    else:
        acc, msle, pre, rec, f1, k, perc = run_NN(first_flg, re_type, processedData, restOfTrain_P)

    result=''
    if (int(perc)==100):
        result = str(round(acc*100,1)) + ';' + str(round(msle,2)) + ';' + str(round(pre*100,1)) + ';' + str(round(rec*100, 1)) + ';' + str(round(f1*100, 1)) + ';' + str(round(k*100, 1)) + ';' + str(round(perc,1)) + ';' + 'Y'
    else:
        result = str(round(acc*100,1)) + ';' + str(round(msle,2)) + ';' + str(round(pre*100,1)) + ';' + str(round(rec*100, 1)) + ';' + str(round(f1*100, 1)) + ';' + str(round(k*100, 1)) + ';' + str(round(perc,1)) + ';' + 'N'

    
    return result

input = sys.argv[1]

input = input.split(',')

trainBatchData = run(input[0], input[1], input[2], input)

print(trainBatchData)
sys.stdout.flush()