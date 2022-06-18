# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
import statistics
  

# Initialize weights
def iniWs():    
    ...
    return()

# Initialize weights for one-layer    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Feed-forward of SNN
def forward():
    ...    
    return() 

#Activation function
def act_function(nodo):
    return(np.maximum(0, nodo))  

# Derivate of the activation funciton
def deriva_act(nodo):
    return (nodo > 0).astype(float) 

#Feed-Backward of SNN
def gradW():    
    ...    
    return()    

# Update Ws
def updW():
    ...    
    return(...)

# Measure
def metricas(x,y):
    cm = confusion_matrix(x,y)
        
    f_score = []
    
    for index, caracteristica in enumerate(cm):
        
        TP = caracteristica[index]
        FP = cm.sum(axis=0)[index] - TP
        FN = cm.sum(axis=1)[index] - TP
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f_score.append(2 * (precision * recall) / (precision + recall))
    f_score.append(statistics.mean(f_score))
    metrics = pd.DataFrame(f_score)
    metrics.to_csv("dl_fscores.csv", index=False, header=False)
    metrics_cm = pd.DataFrame(cm)
    metrics_cm.to_csv("dl_cmatriz.csv", index=False, header=False)
    f_score = np.array(f_score)
    return(cm, f_score)  
    
#Confusion matrix
def confusion_matrix(x,y):
    cm = np.zeros((y.shape[0], x.shape[0]))
    
    for real, predicted in zip(y.T, x.T):
        cm[np.argmax(real)][np.argmax(predicted)] += 1
    return(cm)
#-----------------------------------------------------------------------