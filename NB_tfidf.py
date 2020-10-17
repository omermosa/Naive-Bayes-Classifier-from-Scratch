
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:49:11 2020

@author: OmerMoussa
"""


import pandas as pd
import math as mt
import numpy as np
from scipy.stats import norm

#data load and preprocessing.
vocab=pd.read_csv("vocabulary.txt")["Word"].values

hoc_train=pd.read_csv("hockey_train.txt")
hoc_train.columns=vocab
bas_train=pd.read_csv("baseball_train.txt")
bas_train.columns=vocab

hoc_test=pd.read_csv("hockey_test.txt")
hoc_test.columns=vocab
bas_test=pd.read_csv("baseball_test.txt")
bas_test.columns=vocab

#add label, 1 for baseball, 0 for hockey
#hoc_train["post_type"]=0
#bas_train["post_type"]=1
#
hoc_test["post_type"]=0
bas_test["post_type"]=1

DS=pd.concat([hoc_train,bas_train]) #whole traiing set

Xtest,ytest=pd.concat([hoc_test.iloc[:,:-1],bas_test.iloc[:,:-1]]),pd.concat([hoc_test.iloc[:,-1],bas_test.iloc[:,-1]])
Xtest=Xtest.values
#
#IDF={}
#num_doc=len(finalDS['tweets'])
#for w in freq_dict:
#    IDF[w]=m.log(num_doc/(freq_dict[w]+1))
##get the dictionaries from the dataset (have to be converted from string intdo dicts using literal eval)
#dict_list=[]
#for d in finalDS['dic_words']:
#    dict_list.append(literal_eval(d))
##IDFTF of the whole DS
#for i in range (len(dict_list)):
#    for k in dict_list[i]:
#        dict_list[i][k]*=IDF[k]


#create dict of the classes and their vals to be able to get info
#add label, 1 for baseball, 0 for hockey
train_dict={}
train_dict[0]=(hoc_train.values).astype(float)
train_dict[1]=(bas_train.values).astype(float)

# TF is already provided.
# IDF
def num_doc_with_W(col):
    count=0
    for w in col:
        if w !=0:
            count+=1
    return count
for key, val in  train_dict.items():
    for indx, col in enumerate (zip(*val)):
        count=num_doc_with_W(col)
        #print(indx,count)
        for i in range (len(col)):
            t=train_dict[key][i][indx]
            if t >0:
                #print(t)
                train_dict[key][i][indx]=float(train_dict[key][i][indx])*mt.log(len(col)/(count+1))
                #print( train_dict[key][i][indx], mt.log(len(col)/(count+1)))
    


def mean_std(col):
    #print(sum(col))
    mean=sum(col)/len(col)
    var=sum([(xi-mean)**2 for xi in col])
    var/=(len(col)-1)
    std=mt.sqrt(var)
    return (mean, std)


#get std, mean, count of every col in every class.
summ_dict={}
for key,val in train_dict.items():
    summs=[]
    for col in zip(*val):
        mean,std=mean_std(col)
        summs.append((mean,std,len(col)))
    summ_dict[key]=summs

#get predictions
    
    
def get_maxkey(dic):
    x=max(dic.values())
    #print (dic)
    for k,v in dic.items():
        if v==x:
            return (k,x)
         
def getPredictions(test,summ_dict, class_prob):
    predictions=[]
    learnedProb=[]
    for inst in test:
        probs={}
        am, astd=mean_std(inst)
        for key in summ_dict.keys():
           
            prob=1
            probs[key]=[]
            for w in range (len(inst)):
                summ=summ_dict[key][w]
                if inst[w] >0:
                    m,std=summ[0],summ[1]
                    #print(inst[w],m,std)
                    if std ==0:
                        std=1
                
                    z_score=(inst[w]-m)/std
                    probs[key].append(norm.cdf(z_score)) # p-value of the given prob
            
            for j in range (len(probs[key])):
                prob*=probs[key][j]
            probs[key]=prob
            probs[key]*=class_prob[key] #using form P(y|x)=P(y)*product(P(xi|y))
        print (probs)
        predicted_class, learned_prob=get_maxkey(probs)
        predictions.append(predicted_class)
        learnedProb.append(learned_prob)
    return (learnedProb,predictions)

# building confusion matrix
def buildConfMat(y_predicted,y_test):
    TP,TN,FP,FN=0,0,0,0
   
    yt=np.array(y_test)
    for i in range (len(y_predicted)):
        if(y_predicted[i]==1):
            if(yt[i]==1):
                TP+=1
            else:
                FP+=1
        if(y_predicted[i]==0):
            if(yt[i]==0):
                TN+=1
            else:
                FN+=1    
    
    conf_mat=[[TP,FN],[FP,TN]]        
    print ("Conf Mat ", conf_mat)
    
    Pr=0
    Re=0
    if TP+FP !=0: #avoid undefined values e.g 0/0
        Pr=TP/(TP+FP)
    if TP+FN !=0:
        Re=TP/(TP+FN)
   
    Ac=(TP+TN)/len(y_test)
    print("Precision: ", Pr, "Recall: ", Re, "Accuracy: ", Ac)
    return( Pr,Re,Ac)
    

class_probs={0: len(hoc_train)/len(DS),1:len(bas_train)/len(DS)}

lr, pred=getPredictions(Xtest,summ_dict,class_probs)

print ("Learned from the Training: ", summ_dict)

print ("Learned Prob:", lr)
print ("predicted Class: ", pred)
Pr,Re, Ac=buildConfMat(pred,ytest)














