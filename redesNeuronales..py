import keras
import keras.utils
import numpy as np
import sys
import os
import csv
import time
sys.path.append(os.path.relpath("C:\\Users\\Nelson\\IA\\ia-pc1\\codigo"))
from funciones4 import *

from creaCSV import escribeLinea

from keras import utils as np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


globalList = []

def run_model(train_samples,
              train_labels,              
              porc_num,
              capas,
              unidades_por_capa,
              funcion_activacion):

    cont = 0
    ##Creacion del modelo
    model = Sequential()
    #cada una es una layer
    #primer parametro cantidad nodos, input_shape (por ser primera capa), y funcion de activacion
    model.add(Dense(32, input_shape=(len(train_samples[0]),), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    model.add(Dense(12, activation='relu'))

    while(cont<capas):
        model.add(Dense(unidades_por_capa, activation=funcion_activacion))
        cont = cont+1
    model.add(Dense(19, input_dim=19, activation='relu'))

    #parametros
    #Perdida, optimizador y metrica a usar
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ['accuracy'])

    #Entrenamos el modelo  .fit()
    # 1. muestra
    # 2. labels
    # 3. Batch_size = cuantas muestras queremos que el modelo agrupe cuando este entrenando
    # 4. epochs = cuantas corridas queremos sobre la informacion
    # 5. shuffle = si es true en cada epoch la info estara en orden distinto
    # 6. verbose, la cantidad de veces que imprime por cada corrida sobre la info
    model.summary()
    model.fit(train_samples,train_labels,validation_split=porc_num, batch_size = 20, epochs=100, shuffle = True, verbose = 2)


    #predictions = model.predict(train_samples,batch_size=10,verbose=0)
    predictions = model.predict_classes(train_samples, batch_size=10, verbose=1)
    print(predictions)
    print(len(predictions))
    
    for k in globalList:
        indice = globalList.index(k)
        k.insert(indice,predictions[indice])
        
    

    
def prediccion_r1(muestra_pais,
                  porc_poblacion,
                  porcentaje_num,
                  capas,
                  unidades_por_capa,
                  funcion_activacion):

    #Train samples contiene los votantes con sus atributos
    train_samples = []
    for f in muestra_pais:
        train_samples.append(np.array(f[2:]))
        

    #Train labels tiene por quien voto la persona
    train_labels = []
    for k in muestra_pais:
        train_labels.append(k[0])


    print(train_labels)
    #convierte ambos a numpy
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    #pasa a categorical crossentrophy
    train_labels = keras.utils.to_categorical(train_labels, num_classes=19)

    run_model(train_samples,
              train_labels,
              porcentaje_num,
              capas,
              unidades_por_capa,
              funcion_activacion)
    

def prediccion_r2(muestra_pais,
                  porc_poblacion,
                  porcentaje_num,
                  capas,
                  unidades_por_capa,
                  funcion_activacion):

    #Train samples contiene los votantes con sus atributos
    train_samples = []
    for f in muestra_pais:
        train_samples.append(np.array(f[2:]))

    train_labels = []
    #Train labels tiene por quien voto la persona
    for k in muestra_pais:
        train_labels.append(k[1])

    #convierte ambos a numpy
    train_labels = np.array(train_labels)
    train_samples = np.array(train_samples)
    
    #pasa a categorical crossentrophy
    train_labels = keras.utils.to_categorical(train_labels, num_classes=19)
    
    run_model(train_samples,
              train_labels,
              porcentaje_num,
              capas,
              unidades_por_capa,
              funcion_activacion)


def prediccion_r2_con_r1(muestra_pais,
                  porc_poblacion,
                  porcentaje_num,
                  capas,
                  unidades_por_capa,
                  funcion_activacion):
        
    train_samples = []
    train_labels = []
    #Train samples contiene los votantes con sus atributos
    #Train labels tiene por quien voto la persona
    for f in muestra_pais:
        train_labels.append(f.pop(1))  #f.pop(1)  #le quito el voto de segunda ronda, lo meto en labels
        train_samples.append(np.array(f))

    #convierte ambos a numpy
    train_labels = np.array(train_labels)
    train_samples = np.array(train_samples)
    
    #pasa a categorical crossentrophy
    train_labels = keras.utils.to_categorical(train_labels, num_classes=19)

    run_model(train_samples,
              train_labels,
              porcentaje_num,
              capas,
              unidades_por_capa,
              funcion_activacion)
    

def run_neural_networks(poblacion,
                        porcentaje,
                        capas,
                        unidades_por_capa,
                        funcion_activacion):
    
    #genera muestra
    muestra_pais = generar_muestra_pais(poblacion)

     
    #obtiene el porcentaje de la poblacion
    porc_poblacion = int((porcentaje*poblacion)/100) 
    porcentaje_num = porcentaje/100  #e.g  0.2
    
    print(porc_poblacion)
    print(porcentaje_num)

    
    for l in muestra_pais:
        globalList.append(l[2:])

    cont = 0 
    for k in globalList:
        if(cont < porc_poblacion):
            k.append("False")
        else:
            k.append("True")
        
        cont = cont + 1
    #Run ronda #1    
    prediccion_r1(muestra_pais,
                  porc_poblacion,
                  porcentaje_num,
                  capas,
                  unidades_por_capa,
                  funcion_activacion)

    prediccion_r2(muestra_pais,
                      porc_poblacion,
                      porcentaje_num,
                      capas,
                      unidades_por_capa,
                      funcion_activacion)

    prediccion_r2_con_r1(muestra_pais,
                      porc_poblacion,
                      porcentaje_num,
                      capas,
                      unidades_por_capa,
                      funcion_activacion)


start_time = time.time()
#poblacion #porcentaje #capas #unidades_por_capa #funcion activacion
run_neural_networks(100,20,2,15,"relu")
print("--- %s seconds ---" % (time.time() - start_time))

escribeLinea("mop","Redes",globalList)


##perro= [[3,4,56,7,8],[45,7,9,4,7]]



