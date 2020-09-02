'''
Defination of the cost function correntropy
'''
from keras import backend as K
import numpy as np 

def correntropy(y_true, y_pred): 
    alpha = 2 #shape
    sigma = 0.3 #bandwidth
    lamda = 1 / (np.power(sigma, alpha))
    e = K.abs(y_true - y_pred)
    
    return (1 - K.exp((- lamda) * np.power(e, alpha)))

