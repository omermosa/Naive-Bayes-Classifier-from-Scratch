# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:49:11 2020

@author: OmerMoussa
"""


import pandas as pd
import math as mt
import numpy as np
from scipy.stats import norm

# data processing and reading is here, assume there are inputs for now .
train1=input()
train2=input()
test=input()

def mean_std(col):
    #print(sum(col))
    mean=sum(col)/len(col)
    var=sum([(xi-mean)**2 for xi in col])
    var/=(len(col)-1)
    std=mt.sqrt(var)
    return (mean, std)

#create dict of the classes and their vals to be able to get info
#add label, 1 for baseball, 0 for hockey
train_dict={}
train_dict[0]=train1
train_dict[1]=train2

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
    

class_probs={0: len(train1)/(len(train1)+len(train2)),1:len(train2)/(len(train1)+len(train2))}

lr, pred=getPredictions(test[:][:-1],summ_dict,class_probs)


Pr,Re, Ac=buildConfMat(pred,test[:][-1])














