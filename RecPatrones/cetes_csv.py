# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:26:59 2019

@author: if715029
"""


import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

#%% Lectura de datos
data = pd.read_csv('CETES28.csv',index_col=0)
#%% Crear tabla con todas las tasas de cetes. 
dti = pd.date_range(data.Fecha[0],data.Fecha[514])
Data = pd.DataFrame(index=dti)
Data['CETES28'] = np.zeros(len(dti))
for k in range(len(data.Fecha)-1):
    intervalo = pd.date_range(data.Fecha[k],data.Fecha[k+1])
    Data.loc[intervalo[0:-1]] = data.CETES28[k]




#%%
amxl = pd.read_csv('AMXL.MX.csv')
fechas = amxl['0']

#%% Seleccionar tasa cetes diaria

Cetes_diario =  pd.Series([Data.loc[fechas[i],:].CETES28 for i in range(len(amxl))])
    
    
    
    