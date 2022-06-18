import pandas     as pd
import numpy      as np
import math
from numpy.linalg import norm


def hankel_features(X):
  M = X.shape[0]  
  for i in range(0,M):
      x=X[i,:]
      # ...
      get_features()
  #...
  #...    
  return(Dinput,Doutput) 

# Hankel's features
def get_features():
  ...  
  return(...)


def hankel_svd():
  ...  
  return(...) 


# spectral entropy
def entropy_spectral(X):
  H = 0
  A = saca_a(X)

  for i in range(len(X)):
    p = p(A,i)
    H+= p*np.log2(p)

  H = (-1/np.log2(len(X))) * H
  return H

def saca_a(X):
  A = np.zeros(X.shape)

  for i in range(X.shape):
    Ak = 0
    Ak = np.sum(X*np.exp(-2j*math.pi*(i*np.arange(X.shape/X.shape))))

    Ak = norm(Ak)
    A[i] = Ak
  return A
# Binary Label
# Label binary from raw data 
def Label_binary(y):
    y = pd.get_dummies(y)
    return(y)

# Data norm 
def data_norm(x, xmin, xmax):
    b = 0.99
    a = 0.01
    if xmin == xmax: # En el caso de que de 1/0 se añade un epsilon
        return ((x-xmin)/(xmax-xmin+1e-100))*(b-a)+a
    return ((x-xmin)/(xmax-xmin))*(b-a)+a

# Save Data from  Hankel's features
def save_data_features(Dinp, Dout):
  ...  
  return

# Load data from Data.csv
def load_data(fname):
    data = pd.read_csv(fname, header = None)
    xe = data.iloc[:,:-1]
    xe = np.array(xe)
    ye = data.iloc[:, -1]
    ye = pd.get_dummies(ye)
    ye = np.array(ye)
    ye = ye.T


    return(xe,ye)

# Parameters for pre-proc.
def load_cnf_prep(fname):
    data = pd.read_csv(fname, header = None)
    xe = data.iloc[:data.size]
    return (xe)

# Beginning ...
def main():        
    par_prep    = load_cnf_prep()	
    Data        = load_data()	
    Dinput,Dout = hankel_features(...)
    Dinput      = data_norm(Dinput)
    save_data_features(Dinput,Dout)


if __name__ == '__main__':   
	 main()

