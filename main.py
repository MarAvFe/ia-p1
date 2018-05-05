import argparse
import sys
import os
sys.path.append(os.path.relpath("/home/marcello/Documents/tec/ia/ia-pc1/codigo"))
from funciones4 import *
import matplotlib.pyplot as plt
import time
import datetime

import decisionTree as dt
import knn
#import redesNeuronales as rn
#import regresionLogistica as rl

def trim_sample(sample):
    r1 = []
    r2 = []
    r1_r2 = []
    for i in sample:
        r1.append( [i[0]] + i[2:])
        r2.append( i[1:])
        r1_r2.append( [i[1]] + [i[0]] + i[2:])
    return r1, r2, r1_r2

def usage():
    print("""Usage:
    predict --regresion-logistica [--l1|--l2] |
    predict --red-neuronal [--numero-capas n | --unidades-por-capa u | --funcion-activacion f] |
    predict --arbol [--umbral-poda u]] |
    predict --knn [--k k]
    predict [--prefijo pre] [--poblacion n] [--porcentaje-pruebas]
    predict --h
    """)

def getArgumentos():
	parser = argparse.ArgumentParser()
	#General
	parser.add_argument('--prefijo', nargs = 1, type = str) #archivos # --prefijo <nombre_del_archivo>
	parser.add_argument('--provincia', nargs = 1, type = str)
	parser.add_argument('--poblacion', nargs = 1, type = int, default = 1000) # --poblacion <numero>
	parser.add_argument('--porcentaje_pruebas', nargs = 1, type = int, default = 20) # --porcentaje-pruebas <porcentaje>
	#Regresión Logística
	parser.add_argument('--regresion_logistica', action='store_true')
	parser.add_argument('--l1', action = 'store_true')
	parser.add_argument('--l2', action = 'store_true')
	#Redes Neuronales
	parser.add_argument('--red_neuronal', action = 'store_true')
	parser.add_argument('--numero_capas', nargs = 1, type = int)
	parser.add_argument('--unidades_por_capa', nargs = 1, type = int)
	parser.add_argument('--funcion_activacion', nargs = 1, type = str, default = "relu")
	#Árboles de Decisión
	parser.add_argument('--arbol', action = 'store_true')
	parser.add_argument('--umbral_poda', nargs=1, type=float, default=10)
	#KNN
	parser.add_argument('--knn', action='store_true')
	parser.add_argument('--k', nargs=1, type=int, default=5)
	return parser


def main():
    parser = getArgumentos()
    args = parser.parse_args()

    n = args.poblacion[0]
    print(n)
    if args.regresion_logistica:
        pass
    else:
        if args.provincia:
            dataset = generar_muestra_provincia(n, args.provincia)
        else:
            dataset = generar_muestra_pais(n)
        r1, r2, r1_r2 = trim_sample(dataset)

    if (args.porcentaje_pruebas < 0) or (args.porcentaje_pruebas > 100):
        print("Razon de pruebas inválida:", args.porcentaje_pruebas)
        return
    rate = args.porcentaje_pruebas/100
    print("N:              ",n)
    print("Razon de prueba:", str(int(rate*100)) + "%" )
    print("")
    if args.arbol:
        print("=== Árbol de decision ===")
        resultR1, guessR1 = dt.runAlgorithm(r1, rate)
        resultR2, guessR2 = dt.runAlgorithm(r2, rate)
        resultR1_r2, guessR1_r2 = dt.runAlgorithm(r1_r2, rate)
    elif args.red_neuronal:
        print("=== Red Neuronal ===")
        rn.run_neural_networks(
                            n,
                            rate*100,
                            args.numero_capas,
                            args.unidades_por_capa,
                            args.funcion_activacion)
    elif args.regresion_logistica:
        print("=== Regresión Logística ===")
        regresionLogistica(n, args.l1[0], args.l2[0], rate, args.provincia)
    elif args.knn:
        print("=== KNN ===")
        k = args.k[0]
        resultR1, guessR1 = knn.knnByFoot(r1, rate, int(k))
        resultR2, guessR2 = knn.knnByFoot(r2, rate, int(k))
        resultR1_r2, guessR1_r2 = knn.knnByFoot(r1_r2, rate, int(k))
    else:
        parser.print_help()
        return
    print("\n% Errores:")
    print("Voto 1:", resultR1, "%")
    print("Voto 2:", resultR2, "%")
    print("Voto 2 con voto 1:", resultR1_r2, "%")

    # print(guessR1)
    # print(guessR2)
    # print(guessR1_r2)

if __name__ == '__main__':
    main()

    # BENCH: i = 100
    # BENCH: boop = []
    # BENCH: while (i < 10000):
    # BENCH:     boop.append(int(i))
    # BENCH:     i *= 1.6
    # BENCH: samples = []
    # BENCH: times = []
    # BENCH: for i in boop:
    # BENCH:     print("===== Processing set of " + str(i))
    # BENCH:     print(datetime.datetime.now().strftime("%H:%M:%S"))
    # BENCH:     start = time.time()
    # BENCH:     dataset = generar_muestra_pais(i)
    # BENCH:     r1, r2, r1_r2 = trim_sample(dataset)
    # BENCH:     #print("%s%% de error" % repr(round(dt.runAlgorithm(r1, 0.2))))
    # BENCH:     #print("%s%% de error" % repr(round(dt.runAlgorithm(r2, 0.2))))
    # BENCH:     #print("%s%% de error" % repr(round(dt.runAlgorithm(r1_r2, 0.2))))
    # BENCH:     print("%s%% de error" % repr(round(knn.knnByFoot(r1, 0.2, 2))))
    # BENCH:     #print("%s%% de error" % repr(round(knn.knnByFoot(r2, 0.2, 2))))
    # BENCH:     #print("%s%% de error" % repr(round(knn.knnByFoot(r1_r2, 0.2, 2))))
    # BENCH:     end = time.time()
    # BENCH:     samples.append(i)
    # BENCH:     times.append(end - start)
    # BENCH: print(times)
    # BENCH: print(samples)

    # BENCH: #samples = [20,50,100,500,1000,2500,5000,7500,10000,25000,50000,75000,100000]
    # BENCH: #times = [0.05182814598083496, 0.14002370834350586, 0.24518609046936035, 1.2663624286651611, 2.5744359493255615, 6.702411413192749, 13.350257396697998, 20.115368366241455, 27.97803544998169, 70.38313364982605, 142.0135486125946, 215.7131495475769, 347.14296555519104]
    # BENCH: plt.plot(samples, times, '-bo')
    # BENCH: plt.axis([0, samples[-1]*1.1, 0, times[-1]*1.1])
    # BENCH: plt.xlabel('Muestras')
    # BENCH: plt.ylabel('Duración (s)')
    # BENCH: plt.show()
