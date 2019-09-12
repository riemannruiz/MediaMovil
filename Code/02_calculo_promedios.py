#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Se importa un archivo csv y se calculan los promedios móviles de este.
"""
#%% Importar librerías. 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random 



#%% Importar datos de archivo csv 
stock = ['AMXL.MX','WALMEX.MX','FEMSAUBD.MX','TLEVISACPO.MX','GMEXICOB.MX','GFNORTEO.MX','CEMEXCPO.MX','ALFAA.MX','PE&OLES.MX','GFINBURO.MX','ELEKTRA.MX','MEXCHEM.MX','BIMBOA.MX','AC.MX','KIMBERA.MX','LABB.MX','LIVEPOL1.MX','ASURB.MX','GAPB.MX','ALPEKA.MX','GRUMAB.MX','ALSEA.MX','GCARSOA1.MX','LALAB.MX','IENOVA.MX','PINFRA.MX',]
choice = random.choice(stock)
data = pd.read_csv('../Data/' + choice + '.csv',index_col='Date')
close = data['Adj Close']
#%% Tomar promedios móviles de las ventanas deseadas. 
windows = np.array([ 5, 10, 15, 25, 40, 65])

a = ['']*len(windows)
for i in range(len(a)):
    high = data.High.rolling(windows[i]).mean()
    low = data.Low.rolling(windows[i]).mean()
    adj = data['Adj Close'].rolling(windows[i]).mean()
    a[i] = pd.DataFrame([high,adj,low]).T

#%% Graficar y exportar la gráfica deseada. 
for i in range(len(a)):
    Fig = plt.figure(figsize=(40,5))
    plt.plot(a[i])
    plt.grid()
    plt.plot(close)
    Fig.savefig(('../Imgs/%s%s.png')%choice[i] %i)
    
    
    
    










