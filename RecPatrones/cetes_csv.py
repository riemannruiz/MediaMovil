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
cetes_semanales = pd.read_csv('CETES28.csv',index_col=0)
#%% Crear tabla con todas las tasas de cetes. 
dti = pd.date_range(cetes_semanales.Fecha[0],cetes_semanales.Fecha[514])
cetes_diarios = pd.DataFrame(index=dti)
cetes_diarios['CETES28'] = np.zeros(len(dti))
for k in range(len(cetes_semanales.Fecha)-1):
    intervalo = pd.date_range(cetes_semanales.Fecha[k],cetes_semanales.Fecha[k+1])
    cetes_diarios.loc[intervalo[0:-1]] = (cetes_semanales.CETES28[k]/100+1)**(1/252)-1

cetes_diarios.to_csv('cetes_diarios.csv')
#%% Seleccionar tasa cetes diaria
#amxl = pd.read_csv('AMXL.MX.csv')
#fechas = amxl['0']
#Cetes_diario =  pd.Series([(cetes_diarios.loc[fechas[i],:].CETES28+1)**(1/252)-1 for i in range(len(amxl))])
    
    