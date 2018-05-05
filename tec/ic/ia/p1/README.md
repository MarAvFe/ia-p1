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


### Instalación

Para realizar la instalación de este primer proyecto programado, del curso Inteligencia Artificial, primero deberá de descargar el archivo tec-0.1.zip; y posteriormente, deberá de ejecutar, en línea de comandos, la instrucción que se especifica a continuación:
```
pip install tec-0.1.zip -t <directorio>
```

La variable “directorio” deberá de ser reemplazada por la ruta del sistema  en la que se desea instalar el proyecto:



## Ejecución de pruebas

A continuación se explica cómo se debe de ejecutar cada una de las pruebas de los modelos, para que su ejecución sea exitosa:

* Regresión logística
```
python3 main.py --regresion-logistica --l1 1 --l2 0
```
* Árbol
```
python3 main.py --arbol --umbral  -poda
```
* Red Neuronal
```
python3 main.py --numero-capas 2 --unidades-por-capa 2 --funcion-activacion
```
* KNN
```
python3 main.py --knn --k 15 --poblacion 100
```

## Implementación
En este apartado del documento, se detalla cómo se realizó la implementación de cada uno de los modelos programados.

### Regresión Logística

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

Al ejecutar la prueba con un número pequeño de muestras, la exactitud del modelo disminuye también.



### Árbol de Decisión

El algoritmo se encarga de construir un árbol basado en el set de pruebas que se vaya a utilizar. Este divide a muestra según la razón de pruebas, donde si la razón es un 20%, divide en 2 la muestra, y la primera mitad la divide en 2 para utilizar una parte para entrenamiento y otra para poda.

Por ejemplo si es está utilizando una muestra de 1500 elementos, y la razón de pruebas es de 20%, los elementos del 0-1200 se utilizan para elaboración del modelo y del 1200-1500 para pruebas. De los primeros 1200, el 80% sería de 0-960 para entrenamiento y de 960-1200 para la poda.

### Red Neuronal


Este modelo posee una función principal, que recibe como parámetros la cantidad de votantes que debe de poseer la muestra a generar, el porcentaje de muestra que será utilizado para entrenar al modelo, cantidad de capas que tendrá la red neuronal, las unidades por capas, y por último, la función de activación.

Para confeccionar un predictor de tipo red neuronal, el primer paso a seguir es crear un modelo. En este caso, se creó un modelo de red secuencial, utilizando el parámetro previamente recibido, se le añaden capas al modelo, se configura la cantidad de nodos que tendrá cada capa, y se hace el envío de la función de activación a cada una de estas últimas. Vale resaltar que cada capa puede recibir una función de activación distinta.
Un aspecto importante a tener en cuenta cuando se desarrolla un modelo de este tipo, es que a la primera capa se le debe de hacer saber cuál será el formato de entrada que tendrán los datos; para ello, se usa la función input_shape. Se utiliza la función de pérdida, que en este caso es categorical_crossentropy, de Keras. Finalmente se agrega el optimizador y la métrica de precisión, que en este caso será “accurancy”.
Posteriormente, se procederá a entrenar el modelo. Para ello, se utilizó la función .fit, que recibe un X y una Y. X representa a la lista de votantes (una lista de votantes, en la cuál cada votante incluye todos sus atributos).  La variable Y representa por quién votó esta persona. Recibe dos variables: batch_size (que sirve para agrupar muestras cuando el modelo está entrenando), epoochs que es la cantidad de recorridos que se desea que se ejecuten de una determinada información. La función shuffle mezclará la información después de cada corrida.

Ya una vez el modelo entrenado, se procederá a hacer algunas predicciones. Se utiliza predict para predecir las clases.

Para realizar una predicción de la primera ronda (r1), se toma la muestra total y la divide en una parte para entrenamiento y otra parte para validación. La función train_samples que es X; y train_lables, que representa a Y (por quién votó la persona). Para la primera ronda, a la función train_samples se le quita la etiqueta en el índice 0 (por quién va a votar la persona), y se mantiene el resto de parámetros. Y en train_labels, se toma el valor de por quién votó en la primera ronda dicha persona, y estos datos son los que se suministran al modelo de la red neuronal. La red neuronal recibe numpy arrays, por eso, la información anterior pasará a este formato. Los train_labels (por quién votó la persona), se debe de convertir a “categorical”, pues existen distintas opciones de voto (categorías).

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

## Referencias
* Programa Estado de La Nación (2018). Distribución de juntas receptoras de votos. Recuperado de https://www.estadonacion.or.cr/files/biblioteca_virtual/otras_publicaciones/IndicadoresCantonales_Censos2000y011.xlsx

* Tribunal Supremo de Elecciones República de Costa Rica (2018). 2018 Elecciones Nacionales, Resultados Provinciales. Recuperado de http://resultados2018.tse.go.cr/resultados/#/presidenciales

* Tribunal Supremo de Elecciones República de Costa Rica (2018). 2018 Elecciones Nacionales, Actas de escrutinio. Recuperado de http://www.tse.go.cr/elecciones2018/actas_escrutinio.htm

* Tribunal Supremo de Elecciones República de Costa Rica (2018). Distribución de juntas receptoras de votos. Recuperado de http://www.tse.go.cr/pdf/nacional2018/JRV.pdf
