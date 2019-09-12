# -*- coding: utf-8 -*-
"""
Archivo de importación de datos para Medias móviles. 
"""
#%% Importar librerías. 
from mylib import mylib
import time as _time
import datetime

yahooKeyStats = mylib.yahooKeyStats
#%% Descargar datos históricos. 
#stock = ['AC.MX','ALFAA.MX','ALPEKA.MX','ALSEA.MX','ELEKTRA.MX','IENOVA.MX','MEXCHEM.MX','PE&OLES.MX','PINFRA.MX','WALMEX.MX']
stock = ['AMXL.MX','WALMEX.MX','FEMSAUBD.MX','TLEVISACPO.MX','GMEXICOB.MX','GFNORTEO.MX','CEMEXCPO.MX','ALFAA.MX','PE&OLES.MX','GFINBURO.MX','ELEKTRA.MX','MEXCHEM.MX','BIMBOA.MX','AC.MX','KIMBERA.MX','LABB.MX','LIVEPOL1.MX','ASURB.MX','GAPB.MX','ALPEKA.MX','GRUMAB.MX','ALSEA.MX','GCARSOA1.MX','LALAB.MX','IENOVA.MX','PINFRA.MX',]

today = datetime.date.today()
days =datetime.timedelta(days=2520) #Buscamos 1 año de historia

timestamp=today-days #Solo es para observar que la fecha sea correcta
start = int(_time.mktime(today.timetuple())) #fecha inicial

timestamp2 = datetime.datetime.fromtimestamp(start) #Solo es para observar que la fecha sea correcta
end= int(_time.mktime(timestamp.timetuple())) #fecha final            
#%%

for j in stock:
    data=yahooKeyStats(j,start,end) #descarga los datos de cada ticker
    try:
        data.to_csv(('../Data/%s.csv')%j) #exporta los datos de cada ticker a un csv.
    except: 
        print(j)
    _time.sleep(1)