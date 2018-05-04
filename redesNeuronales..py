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



muestra_pais = generar_muestra_pais(5000)
#analisis(muestra_pais)
#print(muestra_pais)


train_samples = []
train_labels = []

for k in muestra_pais:
    train_labels.append(k[0])
    

train_samples = muestra_pais
for t in train_samples:
    t = np.array(t)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)




def crea_modelo():
    ##Creacion del modelo
    model = Sequential()
    #cada una es una layer
    #primer parametro cantidad nodos, input_shape (por ser primera capa), y funcion de activacion
    #Dense(16, input_dim=19, init='normal', activation='relu'),
    #model.add(Dense(19, input_dim=19, activation='relu'))

    #model.add(Dense(34*8, input_dim=19, init='normal', activation='relu'))
    model.add(Dense(32, input_shape=(19,), activation='sigmoid'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, input_dim=19, activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))
        

    #parametros
    # 1.Optimizador, usamos Adam (se puede usar otras sgd x ejemplo) 
    # 2. perdida. Como la perdida es calculada, (se podria usar tambien mean score error)
    # 3. Metricas. Array de metras que usamos en el modelo
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics = ['accuracy'])


    #Entrenamos el modelo
    #parametros
    # 1. muestra
    # 2. labels
    # 3. Batch_size = cuantas muestras queremos que el modelo agrupe cuando este entrenando
    # 4. epochs = cuantas corridas queremos sobre la informacion
    # 5. shuffle = si es true en cada epoch la info estara en orden distinto
    # 6. verbose, la cantidad de veces que imprime por cada corrida sobre la info
    model.summary()
    model.fit(train_samples,train_labels, validation_split=0.1, batch_size = 100, epochs=50, shuffle = True, verbose = 2)
    







crea_modelo()
print("hola")
