import csv


def escribeLinea(prefijo, nombreModelo, data):
    nombreArchivo = prefijo + '_' + nombreModelo + '.csv'
    with open(nombreArchivo, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

print("creando CSV...")
#data=[["45","68","689","11","68"],["45","68","689","11","68"],["45","68","689","11","68"],["45","68","689","11","68"],["45","68","689","11","68"],["45","68","689","11","68"]]
#escribeLinea("avengers", "redesNeuronales", ro)


