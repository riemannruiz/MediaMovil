class Kclusters:
    def k_clusters(csv, ndias, n_clusters, nombre):
        import pandas as pd
        import numpy as np
        import pickle 
        from sklearn.cluster import KMeans

        # csv = ubicación de los csv de datos
        # ndias = lista de número de días con los cuales se harán los clusters.
        # n_clusters = número de clusters para hacer la partición de cada una de las matrices de ndias
        # nombre = el cual tendrá el archivo .sav generado

        data = []
        for i in csv:
            data.append(pd.read_csv(i, index_col=0))
            
        def crear_ventanas(data,n_ventana):
            n_data = len(data)
            dat_new = np.zeros((n_data-n_ventana+1,n_ventana))
            for k in np.arange(n_ventana):
                dat_new[:,k] = data[k:(n_data-n_ventana+1)+k]
            return dat_new

        vent = []
        for i in ndias:
            close_v = crear_ventanas(data[0]['Close'],i)
            for j in range(1,len(data)):
                close_v = np.concatenate((close_v, crear_ventanas(data[j]['Close'],i)))
            vent.append(close_v)

        cont = len(ndias)       
        for i in range(cont):
            tmp = np.transpose((vent[i].transpose()-vent[i].mean(axis=1))/vent[i].std(axis=1))
            vent[i] = tmp[np.sum(np.isnan(tmp),axis=1)==0]

 
        model_close = []
        for i in range(cont):
            model_close.append(KMeans(n_clusters=n_clusters,init='k-means++').fit(vent[i]))
            
        pickle.dump(model_close,open(nombre+'.sav','wb'))
        
        



class Optimizacion: 
    def crear_ventanas(data,n_ventana):
        import numpy as np
        n_data = len(data)
        dat_new = np.zeros((n_data-n_ventana+1,n_ventana))
        for k in np.arange(n_ventana):
            dat_new[:,k] = data[k:(n_data-n_ventana+1)+k]
        return dat_new

    def portafolio(x,u,p,rcom):
        x_1 = x;
        vp = x[0]+p*x[1] #Valor presente del portafolios
        x_1[0] = x[0]-p*u-rcom*p*abs(u) #Dinero disponible
        x_1[1] = x[1]+u #Acciones disponibles
        return vp,x_1
    
    
    #% Función para realizar la simulación del portafolio
    def portafolio_sim(precio,sit,Ud):
        import numpy as np
        from Simulacion import Optimizacion
        portafolio = Optimizacion.portafolio
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
        
        return Vp
    
    def portafolios_sim(data,sit,Ud):
        import numpy as np
        from Simulacion import Optimizacion
        portafolio_sim = Optimizacion.portafolio_sim
        
        Sim = np.zeros((len(data),len(sit[0])))
        for i in range(len(data)):
            Sim[i] = portafolio_sim(data[i].Close[-len(sit[0]):],sit[i],Ud)
            
        return(Sim)
    
    def simulacion(csv,ndias,model_close,Ud): 
        import numpy as np
        import pandas as pd
        from Simulacion import Optimizacion
        portafolios_sim = Optimizacion.portafolios_sim
        crear_ventanas = Optimizacion.crear_ventanas

        
        # Cargamos bases de datos en .csv
        data = []
        for i in csv: 
            data.append(pd.read_csv(i, index_col=0))
            
        # Creamos ventanas de tiempo
        vent = []
        for j in data: 
            ven = []
            for i in ndias:
                ven.append(crear_ventanas(j['Close'].dropna(),i))  # IMPORTANTE!! Se asume que las bases de datos siempre recibiran el nombre de una columna 'Close'
            vent.append(ven)
    
        # Se estandarizan los datos
        cont = len(ndias)    
        norm = []            
        for j in vent:
            for i in range(cont):
                tmp = j[i].std(axis=1)
                std = np.ones((len(tmp)))
                std[tmp!=0] = tmp[tmp!=0]
                j[i] = np.transpose((j[i].transpose()-j[i].mean(axis=1))/std)
            norm.append(j)
            
        # Se clasifica la situación de los precios en cada cluster de k-means.
        clasif_close = []
        for norm in norm:
            tmp = []
            for i in range(cont):
                tmp.append(model_close[i].predict(norm[i]))
            clasif_close.append(tmp)   
            
        # Cortar la longitud de las clasificaciones para que tengan la misma longitud
        for j in clasif_close:
            for i in range(cont):
                j[i]=j[i][len(norm[0][i])-len(vent[0][-1]):]
            
        # Situación de cada t en T.
        sit = []
        for j in clasif_close:
            s1 = np.zeros(len(j[0]))
            for i in range(cont):
                s1 += j[i]*2**i
            sit.append(s1)
    
        # Simulamos
        Sim = portafolios_sim(data,sit,Ud)
        return(Sim)
    
    
    
    
    
    

class Graficos: 
    def crear_ventanas(data,n_ventana):
        import numpy as np
        n_data = len(data)
        dat_new = np.zeros((n_data-n_ventana+1,n_ventana))
        for k in np.arange(n_ventana):
            dat_new[:,k] = data[k:(n_data-n_ventana+1)+k]
        return dat_new

    def portafolio(x,u,p,rcom):
        x_1 = x;
        vp = x[0]+p*x[1] #Valor presente del portafolios
        x_1[0] = x[0]-p*u-rcom*p*abs(u) #Dinero disponible
        x_1[1] = x[1]+u #Acciones disponibles
        return vp,x_1
    
    
    #% Función para realizar la simulación del portafolio
    def portafolio_sim(precio,sit,Ud):
        import numpy as np
        from Simulacion import Graficos
        portafolio = Graficos.portafolio
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
        
        return T,Vp,X,u

    def portafolios_sim(data,sit,Ud):
        import numpy as np
        from Simulacion import Graficos
        portafolio_sim = Graficos.portafolio_sim
        
        Sim = []
        for i in range(len(data)):
            Sim.append(portafolio_sim(data[i].Close[-len(sit[0]):],sit[i],Ud))
            
        return(np.array(Sim))
    
    def simulacion(csv,ndias,model_close,Ud): 
        import numpy as np
        import pandas as pd
        from Simulacion import Graficos
        import matplotlib.pyplot as plt
        portafolios_sim = Graficos.portafolios_sim
        crear_ventanas = Graficos.crear_ventanas

        
        # Cargamos bases de datos en .csv
        data = []
        for i in csv: 
            data.append(pd.read_csv(i, index_col=0))
            
        # Creamos ventanas de tiempo
        vent = []
        for j in data: 
            ven = []
            for i in ndias:
                ven.append(crear_ventanas(j['Close'].dropna(),i))  # IMPORTANTE!! Se asume que las bases de datos siempre recibiran el nombre de una columna 'Close'
            vent.append(ven)
    
        # Se estandarizan los datos
        cont = len(ndias)   
        norm = []
        for j in vent:
            for i in range(cont):
                tmp = j[i].std(axis=1)
                std = np.ones((len(tmp)))
                std[tmp!=0] = tmp[tmp!=0]
                j[i] = np.transpose((j[i].transpose()-j[i].mean(axis=1))/std)
            norm.append(j)
            
        # Se clasifica la situación de los precios en cada cluster de k-means.
        clasif_close = []
        for norm in norm:
            tmp = []
            for i in range(cont):
                tmp.append(model_close[i].predict(norm[i]))
            clasif_close.append(tmp)   
            
        # Cortar la longitud de las clasificaciones para que tengan la misma longitud
        for j in clasif_close:
            for i in range(cont):
                j[i]=j[i][len(norm[0][i])-len(vent[0][-1]):]
            
        # Situación de cada t en T.
        sit = []
        for j in clasif_close:
            s1 = np.zeros(len(j[0]))
            for i in range(cont):
                s1 += j[i]*2**i
            sit.append(s1)
    
        # Simulamos
        Sim = portafolios_sim(data,sit,Ud)
     
        
        for i in range(len(Sim)): 
            plt.figure(figsize=(8,6))
            plt.subplot(3,1,1)
            plt.plot(Sim[i][0],data[i].Close[-len(sit[0]):])
        #    plt.vlines(1129,data[i].min(),data[i].max())
            plt.ylabel('p(t)')
            plt.grid()
            
            plt.subplot(3,1,2)
            plt.plot(Sim[i][0],Sim[i][1])
            plt.ylabel('vp(t)')
        #    plt.vlines(1129,Sim[i][1].min(),Sim[i][1].max())
            plt.xlabel('time')
            plt.grid()
            
            plt.subplot(3,1,3)
            plt.plot(Sim[i][0],Sim[i][3])
            plt.ylabel('u(t)')
            plt.grid()
            plt.show()
 
        return(Sim)





class Genetico:
    def genetico(func,csv,ndias,model_close,l_vec,n_vec,iteraciones,C,nombre):
        import numpy as np
        from time import time
        import pickle
        
        #func = función a optimizar, esta deberá dar los resultados del vector de decisiones en todas las empresas probadas. 
        #args = argumentos de la función a optimizar.
        #l_vec = longitud de vector de toma de decisiones, en potencias de 2
        #n_vec = cantidad de vectores de toma de decisiones, en potencias de 2.
        #iteraciones = número de ciclos completos que dará el algorítmo genético.
        #C = multiplicador de castigo por desviación estándar
        
        t1 = time()        
        decisiones = np.random.randint(-1,2,(n_vec,l_vec)) # Inicial. 
        
        hist_mean = np.zeros((iteraciones,n_vec//4*5)) # historial de media
        hist_std = np.zeros((iteraciones,n_vec//4*5)) # historial de desviación estandar
        hist_cal = np.zeros((iteraciones,n_vec//4*5)) # historial de calificaciones
        hist_padres = []
        
        punt = np.zeros(n_vec//4*5) # puntuaciones de hijos, se sobre-escribe en cada ciclo
        padres = np.zeros((n_vec//4,l_vec)) # padres, se sobre-escribe en cada ciclo
        
        #Para castigar y premiar baja desviación de rendimientos. 
        pct_mean = np.zeros(punt.shape)
        pct_std = np.zeros(punt.shape)
        
        for cic in range(iteraciones):  
            for i in np.arange(n_vec): ## se simulan todos vectores de decisión para escoger el que de la suma mayor
                
                #######################################################################
                Sim = func(csv,ndias,model_close,decisiones[i]) #########################
                pct = Sim[:,1:]/Sim[:,:-1]-1 ##########################################
                pct = pct.mean(axis=0) ##############################################
                pct_mean[i] = pct.mean() ########################################## todas las empresas
                pct_std[i] = pct.std() ############################################
                #######################################################################
            
            # Se da una calificación a cada vector de toma de decisiones.
            pmr = pct_mean # pct_mean no estandarizado se respalda
            psr = pct_std # pct_std no estandarizado se respalda
            pct_mean = (pct_mean-pct_mean.mean())/pct_mean.std() # pct_mean estandarizado 
            pct_std = (pct_std-pct_std.mean())/pct_std.std() # pct_std estandarizado
            punt = pct_mean-pct_std*C # Se le da una calificación 
            
            # Se escogen los padres.
            decisiones = np.concatenate((decisiones,padres)) # agregamos los 'padres' de las nuevas generaciones a la lista. 
            padres = decisiones[np.argsort(punt)[-int(n_vec//4):]] # se escojen los padres
            pct_mean[-int(n_vec//4):] = pmr[np.argsort(punt)[-int(n_vec//4):]] # se guarda la media que obtuvieron los padres  
            pct_std[-int(n_vec//4):] = psr[np.argsort(punt)[-int(n_vec//4):]] # se guarda la desviación que obtuvieron los padres 
            
            hist_mean[cic,:] = pmr #se almacena el promedio de los padres para observar avance generacional
            hist_std[cic,:] = psr
            hist_cal[cic,:] = punt
            
            # Se mutan los vectores de toma de decisiones
            decisiones = np.array([[np.random.choice(padres.T[i]) for i in range(l_vec)] for i in range(n_vec)])
            for k in range(n_vec): ## mutamos la cuarta parte de los dígitos de los n_vec vectores que tenemos. 
                for i in range(int(l_vec//4)):
                    decisiones[k][np.random.randint(0,l_vec)] = np.random.randint(0,3)-1
                  
            # Para imprimir el proceso del algoritmo genérico en relación al total por simular.    
            print(np.ceil((1+cic)/iteraciones*1000)/10)
        
            # Cada 10 iteraciones se guardan los resultados de las simulaciones en un respaldo. 
            if cic % 10 == 0: 
                hist_padres.append(padres)
                pickle.dump([padres,hist_mean,hist_std,hist_cal,hist_padres],open('tmp.sav','wb'))
            
        print(padres)
        print('tiempo de ejecución en seg.:')
        print(time()-t1)
        
        pickle.dump([punt,padres,hist_mean,hist_std,hist_cal,hist_padres],open(nombre + '.sav','wb')) # guarda las variables más importantes al finalizar. 
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        