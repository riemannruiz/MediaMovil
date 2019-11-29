#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graficar rendimientos de los padres en periodos desconocidos.
"""
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulacion import Optimizacion
from Simulacion import Graficos
from Simulacion import Genetico
from Simulacion import Kclusters


simulacion = Optimizacion.simulacion
grafico = Graficos.simulacion
genetico = Genetico.genetico
k_clusters = Kclusters.k_clusters

#%% Datos en csv
csv = ['AMXL.MX','WALMEX.MX','TLEVISACPO.MX','GMEXICOB.MX','GFNORTEO.MX','CEMEXCPO.MX','PENOLES.MX','GFINBURO.MX','ELEKTRA.MX','BIMBOA.MX','AC.MX','KIMBERA.MX','LABB.MX','LIVEPOL1.MX','ASURB.MX','GAPB.MX','ALPEKA.MX','GRUMAB.MX','ALSEA.MX','GCARSOA1.MX','PINFRA.MX']
for i in np.arange(len(csv)):
    csv[i] = '../Test/%s.csv'%csv[i] #Se utilizan todos los datos para hacer las pruebas
cetes = 'cetes_diarios.csv'
ndias = [5,20,40,125]
n_clusters = 4
nombre = 'model_close'
model_close = pickle.load(open('model_close.sav','rb'))

#%%
nombre = 'Intento3'
[punt,padres,hist_mean,hist_std,hist_cal,hist_padres] = pickle.load(open(nombre + '.sav','rb'))

#%%
""" Trabajaremos con los Padres históricos que se generaron cada 10 iteraciones (20 it. en total). """
#%%
hm_padres = np.zeros((hist_padres[0].shape[0]*len(hist_padres),hist_padres[0].shape[1])) # Creamos la matriz para construir el mapa de calor. 
for i in np.arange(len(hist_padres)):
    hm_padres[i*hist_padres[0].shape[0]:(i+1)*hist_padres[0].shape[0],:] = hist_padres[i]

#%% Dibujamos el heatmap
fig = plt.figure(figsize=(12,12))
plt.imshow(hm_padres)

#%% Segundo heatmap, esta vez con el promedio de los padres. 
hm_padres2 = np.zeros((len(hist_padres),hist_padres[0].shape[1])) # Creamos la matriz para construir el mapa de calor. 
for i in np.arange(len(hist_padres)):
    hm_padres2[i,:] = np.round(hist_padres[i].mean(axis=0))
#    hm_padres2[i,:] = hist_padres[i].mean(axis=0)
    
#%% Dibujamos el heatmap
fig = plt.figure(figsize=(24,4))
plt.imshow(hm_padres2)

#%%############################################################################
#################### Tiempos desconocidos. Mismos padres. #####################
ult = 176
hist_padres[10]
rf = 0.0471/252
gen = 5

#%%

Vals = np.zeros((len(hist_padres[0])*gen,6))
cont = 0 
for i in np.arange(gen)+1: 
    for j in range(len(hist_padres[0])):
        Vp = simulacion(csv,ndias,model_close,hist_padres[-i][-j],cetes)
        plt.plot(Vp[:,-ult:-1].T/Vp[:,-ult].T) # Grafica el comportamiento de nuestro padre en cada uno de los activos
        plt.plot(np.mean(Vp[:,-ult:-1].T/Vp[:,-ult].T,axis=1)) #Grafica el comportamiento de nuestro padre en todo el portafolio

        pct =  (Vp[:,1:]/Vp[:,0:-1]-1)
        g_pct =  pct.mean(axis=0)
        
        mean1 = g_pct[:-ult].mean()
        mean2 = g_pct[-ult:].mean()
        std1 = g_pct[:-ult].std()
        std2 = g_pct[-ult:].std()
        shpe1 = (mean1-rf)/std1
        shpe2 = (mean2-rf)/std2
        Vals[cont,:] = [mean1,mean2,std1,std2,shpe1,shpe2]
        cont += 1
        
        



























