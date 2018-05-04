import decisionTree as dt
import sys
import os
sys.path.append(os.path.relpath("/home/marcello/Documents/tec/ia/ia-pc1/codigo"))
from funciones4 import *
import matplotlib.pyplot as plt
import time

def trim_sample(sample):
    r1 = []
    r2 = []
    r1_r2 = []
    for i in sample:
        r1.append( [i[0]] + i[2:])
        r2.append( i[1:])
        r1_r2.append( [i[1]] + [i[0]] + i[2:])
    return r1, r2, r1_r2

if __name__ == '__main__':
    dataset = generar_muestra_pais(i)
    r1, r2, r1_r2 = trim_sample(dataset)
    
    print("%s%% de error" % repr(round(dt.runAlgorithm(r1),2)))
    print("%s%% de error" % repr(round(dt.runAlgorithm(r2),2)))
    print("%s%% de error" % repr(round(dt.runAlgorithm(r1_r2),2)))


    # BENCH: samples = []
    # BENCH: times = []
    # BENCH: for i in [20,50,100,500,1000,2500,5000,7500,10000,25000,50000,75000,100000,500000,1000000]:
    # BENCH:     print("===== Processing set of " + str(i))
    # BENCH:     start = time.time()
    # BENCH:     dataset = generar_muestra_pais(i)
    # BENCH:     r1, r2, r1_r2 = trim_sample(dataset)
    # BENCH:     print("%s%% de error" % repr(round(dt.runAlgorithm(r1),2)))
    # BENCH:     print("%s%% de error" % repr(round(dt.runAlgorithm(r2),2)))
    # BENCH:     print("%s%% de error" % repr(round(dt.runAlgorithm(r1_r2),2)))
    # BENCH:     end = time.time()
    # BENCH:     samples.append(i)
    # BENCH:     times.append(end - start)
    # BENCH: print(times)

    # BENCH: samples = [20,50,100,500,1000,2500,5000,7500,10000,25000,50000,75000,100000]
    # BENCH: times = [0.05182814598083496, 0.14002370834350586, 0.24518609046936035, 1.2663624286651611, 2.5744359493255615, 6.702411413192749, 13.350257396697998, 20.115368366241455, 27.97803544998169, 70.38313364982605, 142.0135486125946, 215.7131495475769, 347.14296555519104]
    # BENCH: plt.plot(samples, times, '-bo')
    # BENCH: plt.axis([0, samples[-1]*1.1, 0, times[-1]*1.1])
    # BENCH: plt.ylabel('Muestras')
    # BENCH: plt.xlabel('Duración (s)')
    # BENCH: plt.show()
