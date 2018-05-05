import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as skl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from simuladorDeVotantes import *
import csv

def decode(datum):
	return np.argmax(datum)



def generarMuestra(provincia, tamanoPoblacion):
	nombresColumnas = ["P_RONDA","S_RONDA","CANTON","SEXO","EDAD","DENSIDAD","DEPENDIENTE","PROM_OCUPANTES","VIVIENDA","HACINAMIENTO",
	"ANALFABETISMO","ANNOS_APROBADOS","PORC_EDUCACION_F","DESEMPLEADO","PARTICIPACION_ECONOMICA","ASEGURADO","EXTRANJERO",
	"DISCAPACITADO","JEFATURA_COMPARTIDA"]
	tmpMuestra=[]
	cont = 0
	if provincia != "":
		#Se genera la muestra del país
		muestra = generar_muestra_provincia(int (tamanoPoblacion*1.3), provincia)
	else:
		#Se genera la muestra de la provincia
		muestra = generar_muestra_pais(int(tamanoPoblacion*1.3))
	tmpMuestra.insert(0, nombresColumnas)
	for votante in muestra:
		hayNegativos=False
		for atributo in votante:
			if atributo ==-1:
				hayNegativos=True
		if hayNegativos==False:
			if (cont<tamanoPoblacion):
				cont=cont+1
				tmpMuestra.append(votante)
			else:
				return tmpMuestra
	return tmpMuestra


def regresionLogistica(tamanoPoblacion=1000, l1=0, l2=0, porcentajePruebas=0.1,  provincia=""):
	tasaAprendizaje = 0.1
	num_epochs = 20
	display_step = 1
	perdida=0
	#Genera votantes y crea un Pandas.DataFrame
	df=pd.DataFrame(generarMuestra(provincia, tamanoPoblacion))
	df =df.fillna(value=0.0) # Quita los NAN
	df.to_csv('tmp.csv', index=False, header=False)#guardar csv
	df = pd.read_csv('tmp.csv')
	df =df.fillna(value=0.0)

	# Características
	x_listaColumnas_r1 =["CANTON","SEXO","EDAD","DENSIDAD","DEPENDIENTE","PROM_OCUPANTES","VIVIENDA","HACINAMIENTO",
	"ANALFABETISMO","ANNOS_APROBADOS","PORC_EDUCACION_F","DESEMPLEADO","PARTICIPACION_ECONOMICA","ASEGURADO","EXTRANJERO","DISCAPACITADO","JEFATURA_COMPARTIDA"]
	y_listaColumnas_r1=['P_RONDA']

	x_prediccion_r1 = df.loc[:, x_listaColumnas_r1]
	y_prediccion_r1 = df.loc[:, y_listaColumnas_r1]

	# Codificar
	oneHot = OneHotEncoder() # Declaración OneHotEncoder
	oneHot.fit(x_prediccion_r1)
	x_prediccion_r1 = oneHot.transform(x_prediccion_r1).toarray()
	oneHot.fit(y_prediccion_r1)
	y_prediccion_r1 = oneHot.transform(y_prediccion_r1).toarray()

	# Divide la muestra en datos de prueba y entrenamiento
	x_r1_entrenamiento, x_r1_prueba, y_r1_entrenamiento, y_r1_prueba = train_test_split(x_prediccion_r1, y_prediccion_r1, test_size = porcentajePruebas, random_state=0)
	#tamano parámetros
	nCaracteristicas = x_r1_entrenamiento.shape[1]
	nEtiquetas= y_r1_entrenamiento.shape[1]
	if l1 != 0:
		regularizer = tf.contrib.layers.l1_regularizer(l1)
	elif l2 != 0:
		regularizer = tf.contrib.layers.l2_regularizer(l2)
	with tf.name_scope("Declaring_placeholder"):
		# X características
		x_r1 = tf.placeholder(tf.float32, shape=[None, nCaracteristicas])
		# y etiquetas
		#y_r1 = tf.placeholder(tf.float32, shape=[None, nEtiquetas])
		y_r1 = tf.placeholder(tf.float32, shape=[None, nEtiquetas])

	with tf.name_scope("Declaring_variables"):
		W_r1 = tf.Variable(tf.zeros([nCaracteristicas, nEtiquetas]))
		b_r1 = tf.Variable(tf.zeros([nEtiquetas]))

	with tf.name_scope("Declaring_functions"):
		#y_ = tf.nn.softmax(tf.add(tf.matmul(X, W), b))#prediction
		prediccion_r1 = tf.nn.softmax(tf.matmul(x_r1, W_r1) + b_r1)
		prediction_is_correct = tf.cast(tf.equal(tf.argmax(prediccion_r1, 1), tf.argmax(y_r1, 1)), tf.float32)
	with tf.name_scope("calculating_cost"):
		#costo_r1 = tf.nn.softmax_cross_entropy_with_logits(labels=y_r1, logits=prediccion_r1)
		costo_r1 = tf.reduce_mean(-tf.reduce_sum(y_r1 * tf.log(prediccion_r1), axis=1))

	with tf.name_scope("declaring_gradient_descent"):
		optimizador_r1 = tf.train.GradientDescentOptimizer(learning_rate = tasaAprendizaje).minimize(costo_r1)
		#optimizador_r1 = tf.train.FtrlOptimizer(learning_rate=tasaAprendizaje, l1_regularization_strength=1, l2_regularization_strength=0)
		#optimizer=tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1.0, l2_regularization_strength=1.0)
	with tf.name_scope("starting_tensorflow_session"):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			#inicia el entrenamiento
			for epoch in range(num_epochs):
				perdida = 0
				_, c = sess.run([optimizador_r1, costo_r1], feed_dict = {x_r1: x_r1_entrenamiento, y_r1: y_r1_entrenamiento})
				perdida += c
				#sess.run([optimizer], feed_dict={X: X_train, y: y_train})
				#imprimir el costo del entrenamiento
				if (epoch+1) % display_step == 0:

					print("Epoch: {}".format(epoch + 1), "loss={}".format(perdida))
			#W_hat, b_hat = sess.run([W, b])
			#print("w_hat",W_hat)
			#print("b_hat",b_hat)

			# Probar el modelo
			predicted_class = tf.greater(prediccion_r1,0.5)
			correct = tf.equal(predicted_class, tf.equal(y_r1,1.0))
			accuracy = tf.reduce_mean( tf.cast(correct, 'float') )
			print("Exactitud: ", accuracy.eval({x_r1: x_r1_entrenamiento, y_r1: y_r1_entrenamiento}))
			print("Pérdida: ", perdida)
			#prediction=tf.argmax(y,1)
			#print ("predictions", prediction.eval(feed_dict={X: X_test}))
    return 100-accuracy
#regresionLogistica()
