class mylib:
    def yahooKeyStats(stock,start,end):
        import time as _time
        import datetime
        import requests
        import pandas as pd
        import re
        import string
        try:
            url='https://query1.finance.yahoo.com/v7/finance/download/'+stock+'?period1='+str(end)+'&period2='+str(start)+'&interval=1d&events=history&crumb=p46S32cqcsI'
            files = {'file': ('report.csv', 'some,data,to,send\nanother,row,to,send\n')}
            r = requests.post(url, files=files)
            lines = [re.sub('\s+' ,' ' , line.strip() , 1) for line in r.text.split('\n')]
            S = pd.Series(lines)
            data = S.str.split(',').tolist()
            data=data[:-1]
            data=pd.DataFrame(data)
            header=data.iloc[0,:]
            index=data.iloc[:,0]
            data=pd.DataFrame(data.iloc[1:,1:])
            data.columns=header[1:]
            data.index=index[1:]
            return data
        except Exception:
            print (stock + ' not found')
            return -1
    #% Funcion del modelo del portafolio
    def portafolio(x,u,p,rcom):
        x_1 = x;
        vp = x[0]+p*x[1] #Valor presente del portafolios
        x_1[0] = x[0]-p*u-rcom*p*abs(u) #Dinero disponible
        x_1[1] = x[1]+u #Acciones disponibles
        return vp,x_1
    
    
    #% Función para realizar la simulación del portafolio
    def portafolio_sim(precio,sit,Ud):
        import numpy as np
        from mylib import mylib
        portafolio = mylib.portafolio
        T = np.arange(len(precio))
            
        Vp = np.zeros(T.shape)
        X  = np.zeros((T.shape[0]+1,2)) 
        u = np.zeros(T.shape)
        X[0][0] = 10000
        rcom = 0.0025
        
        for t in T:
            
            u_max = np.floor(X[t][0]/((1+rcom)*precio[t])) # Numero maximo de la operacion
            u_min  = X[t][1] # Numero minimo de la operacion
            
            #AC (operacion matricial)
            if Ud[int(sit[t])]>0:
                u[t] = u_max*Ud[int(sit[t])]
            else:
                u[t] = u_min*Ud[int(sit[t])]
            
            Vp[t],X[t+1]=portafolio(X[t],u[t],precio[t],rcom)
        
    #    return T,Vp,X,u
        return Vp
    
    def crear_ventanas(data,n_ventana):
        import numpy as np
        n_data = len(data)
        dat_new = np.zeros((n_data-n_ventana+1,n_ventana))  
        for k in np.arange(n_ventana):
            dat_new[:,k] = data[k:(n_data-n_ventana+1)+k]
        return dat_new
    
    def Sit(close,ndias,model_close):
        import numpy as np
        #close es una serie de tiempo de numpy que contiene los precios de cierre de acción deseada. 
        #ndias es un vector tipo '[5,20,40,125]' que da situaciones de 
        
        
        # Crear la ventanas para el analisis
        vent = []
        for i in ndias:
            vent.append(mylib.crear_ventanas(close,i))
        
        
        # Normalizar las ventas
        cont = len(ndias)    
        for i in range(cont):
            vent[i] = np.transpose((vent[i].transpose()-vent[i].mean(axis=1))/vent[i].std(axis=1))
            
        # Clasificacion de las ventanas usando el modelo KMeans entrenado
        clasif_close = []
        for i in range(cont):
            clasif_close.append(model_close[i].predict(vent[i]))
        
        # Elegir el mismo numero de clasificaciones para las diferentes ventanas
        for i in range(cont):
            clasif_close[i]=clasif_close[i][len(vent[i])-len(vent[-1]):]
            
        # Crear el ventor de situaciones
        sit = np.zeros(len(clasif_close[0]))
        for i in range(cont):
            sit += clasif_close[i]*4**i
        
        # Tomar los precios que se utilizaran para la simulacion
        precio = close[-len(sit):]
        
        return (precio,sit)
