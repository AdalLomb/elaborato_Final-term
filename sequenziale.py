import numpy as np
import time

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def mean_shift(X, radius):
    # Inizializzazione dei centroidi
    centroids = X.copy()
    # Numero di dati di input
    m = X.shape[0]
    # Lista per tenere traccia delle etichette di clustering
    labels = np.zeros(m)
    # Ciclo fino alla convergenza dei centroidi
    while True:
        # Lista per tenere traccia dei nuovi centroidi
        new_centroids = []
        # Ciclo su tutti i centroidi attuali
        for i, centroid in enumerate(centroids):
            # Lista per tenere traccia dei punti all'interno del raggio di banda
            within_radius = []
            # Ciclo su tutti i dati di input
            for j, point in enumerate(X):
                # Calcola la distanza euclidea tra il punto e il centroide
                distance = euclidean_distance(point, centroid)
                # Se il punto e' all'interno del raggio di banda, aggiungilo alla lista within_radius
                if distance <= radius:
                    within_radius.append(j)
            # Calcola il nuovo centroide come la media dei punti all'interno del raggio di banda
            if within_radius:
                new_centroid = np.mean(X[within_radius], axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroid)
        # Convergenza raggiunta se i centroidi non cambiano
        if np.allclose(np.array(new_centroids), np.array(centroids)):
            break
        centroids = new_centroids.copy()
    # Assegna un'etichetta di clustering a ciascun dato di input in base al centroide piu' vicino
    for i, point in enumerate(X):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        labels[i] = cluster
    return labels, centroids

# Legge file di input 
with open("sample_1000.txt", "r") as file:
        # legge l'intera riga dal file
    data  = file.read().split(',')
data = [float(d.strip()) for d in data]

X = np.array(data).reshape(-1,2)

# Inizializzazione del raggio di banda
radius = 0.01

# Calcolo del tempo di esecuzione
start_time = time.time()

# Effettua il clustering
labels, centroids = mean_shift(X, radius)

# Calcolo del tempo di esecuzione
end_time = time.time()
elapsed_time = end_time - start_time

# Stampa i risultati
print("Tempo di elaborazione: ", elapsed_time)
