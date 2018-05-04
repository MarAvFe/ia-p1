import keras
import numpy as np
import sys
import os
sys.path.append(os.path.relpath("C:\\Users\\Nelson\\IA\\ia-pc1\\codigo"))
from funciones4 import *

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
    model.add(Dense(32, input_shape=(len(train_samples[0]),), activation='sigmoid'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    model.add(Dense(12, activation='relu'))

    while(cont<capas):
        model.add(Dense(unidades_por_capa, activation=funcion_activacion))
        cont = cont+1
    model.add(Dense(1, input_dim=19, activation='relu'))

    #parametros
    #Perdida, optimizador y metrica a usar
    model.compile(loss='mean_squared_error',optimizer='adam', metrics = ['accuracy'])

    #Entrenamos el modelo  .fit()
    # 1. muestra
    # 2. labels
    # 3. Batch_size = cuantas muestras queremos que el modelo agrupe cuando este entrenando
    # 4. epochs = cuantas corridas queremos sobre la informacion
    # 5. shuffle = si es true en cada epoch la info estara en orden distinto
    # 6. verbose, la cantidad de veces que imprime por cada corrida sobre la info
    model.summary()
    model.fit(train_samples,train_labels, validation_split=porc_num, batch_size = 100, epochs=300, shuffle = True, verbose = 2)

    
def prediccion_r1(muestra_pais,
                  porc_poblacion,
                  porcentaje_num,
                  capas,
                  unidades_por_capa,
                  funcion_activacion):

    train_samples = []
    train_labels = []
    
    #llena labels
    for k in muestra_pais:
        train_labels.append(k[0])

    #llena samples y convierte a numpy cada lista
    train_samples = muestra_pais
    for t in train_samples:
        t = np.array(t)

    #convierte ambos a numpy
    train_labels = np.array(train_labels)
    train_samples = np.array(train_samples)

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

    train_samples_aux=[]
    train_samples = []
    train_labels = []
    
    #llena labels
    for k in muestra_pais:
        train_labels.append(k[1])

    #llena samples y convierte a numpy cada lista
    train_samples_aux = muestra_pais
    for t in train_samples_aux:
        lista_sin_r1 = t[1:]
        train_samples.append(np.array(lista_sin_r1))


    #convierte ambos a nclsumpy
    train_labels = np.array(train_labels)
    train_samples = np.array(train_samples)

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
    
    #llena labels
    for k in muestra_pais:
        train_labels.append(k[1])

    #llena samples y convierte a numpy cada lista
    train_samples = muestra_pais
    for t in train_samples:
        t = np.array(t)

    #convierte ambos a numpy
    train_labels = np.array(train_labels)
    train_samples = np.array(train_samples)

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




#poblacion #porcentaje #capas #unidades_por_capa #funcion activacion
run_neural_networks(7000,20,2,10,"softmax")

