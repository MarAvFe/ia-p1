# Proyecto 1 - IA

El objetivo principal de desarrollar este proyecto, es que poner en práctica los conocimientos adquiridos en el curso Inteligencia Artificial, impartido en el Tecnológico de Costa Rica. Motivo por el cual, se desarrolló un proyecto clasificador de datos.

Este proyecto implementa cuatro algoritmos diferentes para clasificar poblaciones aleatorias basadas en datos reales del censo del 2011 y las votaciones de abril y mayo 2018. Los algoritmos utilizados son k-vecinos más cercanos, redes neuronales, regresión logística y árboles de decisión.

## Cómo empezar

Estas instrucciones detallan cómo conseguir una copia funcional del proyecto en una máquina local, para propósitos de desarollo y pruebas.

Deberá de descargar el programa, en la siguiente dirección:
```
https://github.com/MarAvFe/ia-p1
```

### Requisitos

En esta sección del documento, se detallan los requisitos mínimos necesarios para poder ejecutar, de manera exitosa, el programa:

1. Una computadora.
2. Conexión a internet (para descargar librerías).

#### Tensorflow

1. La guía oficial se puede encontran en [este link](https://www.tensorflow.org/install/install_linux)
2. Verificar la instalación de python3
```
$ sudo apt-get install python3-pip python3-dev
```

3. Instalar tensorflow por medio de pip
```
$ pip install --upgrade tensorflow
```

4. Probar la instalación con un pequeño programa

    ```python
    # Python: testTensorflow.py
    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))

    ## $ python3 testTensorflow.py
    ## Output: Hello, TensorFlow
    ```

#### Keras
1. Antes de instalar Keras, deberá de tener instalado TensorFlow.
2. Ejecutar en línea de comandos, la siguiente instrucción:
```
sudo pip install keras
```

#### Pandas
1. Ejecutar en línea de comandos, la siguiente instrucción:
```
pip install pandas
```

#### Sklearn
1. Ejecutar en línea de comandos, la siguiente instrucción:
```
pip install -U scikit-learn
```

#### El generador de pc1

### Instalación

Para realizar la instalación de este primer proyecto programado, del curso Inteligencia Artificial, primero deberá de descargar el archivo tec-0.1.zip; y posteriormente, deberá de ejecutar, en línea de comandos, la instrucción que se especifica a continuación:
```
pip install tec-0.1.zip -t <directorio>
```

La variable “directorio” deberá de ser reemplazada por la ruta del sistema  en la que se desea instalar el proyecto:



## Ejecutar las pruebas

A continuación se explica cómo se debe de ejecutar cada una de las pruebas de los modelos, para que su ejecución sea exitosa:

```
Give an example
```


## Implementación
En este apartado del documento, se detalla cómo se realizó la implementación de cada uno de los modelos programados.

### Regresión Logística
##### Implementación

Para la implementación de la regresión lineal, se utilizó la biblioteca Pandas, para el modelado de la información. Pandas es una librería que proporciona estructuras de datos flexibles, que permiten trabajar con dichos datos de manera eficiente.
Inicialmente, utilizando el generador desarrollado previamente, se obtiene la información de una determinada cantidad de votantes (muestra).

Utilizando Pandas, se coloca toda esta información en un DataFrame.
Posteriormente, se separan los datos: Las columnas que contienen las características o features, serán tomadas como X de la función; las columnas etiquetas o labels, serán tomanas como Y de la función. En este caso, se tomó como label, los datos de la primera ronda y segunda ronda (según la regresión solicitada por el usuario, ya sea r1, r2 o r2_con_r1).

Posteriormente, se convirtieron a binarios los datos, para ello se utilizó la librería Sklearn, que contiene la función OneHotEncoder.

Adicional a esto, se realizó la división de datos: Se separan los datos para entrenamiento y para prueba, según un parámetro enviado por el usuario. Para ello se utilizó la función train_test_split.

Para la regularización, se utilizó TensorFlow, y sus funciones para aplicar l1 y l2 a la solución.

Se utilizó TensorFlow  para generar el modelo: se parametrizaron las variables, placeholder, se hizo la función de pérdida, el optimizador (utilizando GradientDescent).

Con una tasa de aprendizaje, se procede a entrenar al modelo, utilizando ciclos o epochs, para lo cual se utilizan los datos de entrenamiento previamente mencionados.

También se procede a obtener la exactitud o accuracy del modelo.

Por último se utiliza una función para predecir el resultado de la votación, obtenida, según un conjunto de datos, que describen a un determinado votante. Para ello se utilizó el  modelo que fue previamente entrenado.

##### Conclusiones
* Al ejecutar la prueba con un número pequeño de muestras, la exactitud del modelo disminuye también.
 


### Árbol de Decisión

Cómo se hizo...

### Red Neuronal

Cómo se hizo...

### KNN

Cómo se hizo...

## Construido con

* [Tensorflow](http://www.dropwizard.io/1.0.2/docs/) - TODO: The web framework used
* [Keras](https://maven.apache.org/) - TODO: Dependency Management
* [matplotlib](https://matplotlib.org/) - Usado para graficar los resultados

## Autores

* **Marcello Ávila** - *Desarrollador*
* **Stefi Falcón** - *Desarrollador*
* **Nelson Gómez** - *Desarrollador*

## Licencia

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Reconocimientos

* [Decision Tree Classifier from Scratch](https://github.com/random-forests/tutorials/blob/master/decision_tree.py)
* Hat tip to anyone who's code was used
* Inspiration
* etc

## Referencias
* Programa Estado de La Nación (2018). Distribución de juntas receptoras de votos. Recuperado de https://www.estadonacion.or.cr/files/biblioteca_virtual/otras_publicaciones/IndicadoresCantonales_Censos2000y011.xlsx

* Tribunal Supremo de Elecciones República de Costa Rica (2018). 2018 Elecciones Nacionales, Resultados Provinciales. Recuperado de http://resultados2018.tse.go.cr/resultados/#/presidenciales

* Tribunal Supremo de Elecciones República de Costa Rica (2018). 2018 Elecciones Nacionales, Actas de escrutinio. Recuperado de http://www.tse.go.cr/elecciones2018/actas_escrutinio.htm

* Tribunal Supremo de Elecciones República de Costa Rica (2018). Distribución de juntas receptoras de votos. Recuperado de http://www.tse.go.cr/pdf/nacional2018/JRV.pdf
