# Proyecto 1 - IA

Este proyecto implementa 4 algoritmos diferentes para clasificar poblaciones aleatorias basadas en datos reales del censo del 2011 y las votaciones de abril y mayo 2018. Los algoritmos utilizados son k-vecinos más cercanos, redes neuronales, regresión y árboles de decisión.

## Cómo empezar

Estas instrucciones detallan cómo conseguir una copia funcional del proyecto en una máquina local para propósitos de desarollo y pruebas.

### Requisitos

Qué cosas se necesitan para instalar el proyecto y cómo hacerlo
What things you need to install the software and how to install them

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

#### Pandas

#### El generador de pc1

### Instalación

TODO: Aqui yo creo que es como "diay, copie los archivos y ya.". Verifique las rutas de los archivos del genreador así y así.

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Ejecutar las pruebas

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Implementación

### Regresión

Cómo se hizo...

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
