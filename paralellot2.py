import asyncio
import numpy as np
from sklearn.cluster import MeanShift
import time

async def mean_shift_async(X):
    ms = MeanShift()
    ms.fit(X)
    return ms.labels_, ms.cluster_centers_

async def run_mean_shift_parallel(X, n_jobs):
    # Suddivide i dati in n_jobs blocchi
    blocks = np.array_split(X, n_jobs)
    
    # Crea un task per ogni blocco e avvia l'esecuzione parallela
    tasks = [asyncio.create_task(mean_shift_async(block)) for block in blocks]
    results = await asyncio.gather(*tasks)
    
    # Unisce i risultati dei singoli blocchi
    labels = np.concatenate([r[0] for r in results])
    centers = np.concatenate([r[1] for r in results])
    
    return labels, centers

with open("sample_20000.txt", "r") as file:
        # legge l'intera riga dal file
    data  = file.read().split(',')
data = [float(d.strip()) for d in data]

X = np.array(data).reshape(-1,2)

start_time = time.time()
# Esegue il mean-shift clustering in parallelo su n_jobs processi
labels, centers = asyncio.run(run_mean_shift_parallel(X, n_jobs=2))
end_time = time.time()

# Stampa i risultati
print("Numero di cluster:", len(np.unique(labels)))
print("Centri dei cluster:\n", centers)
print(f"Tempo di elaborazione parallelo: {end_time - start_time:.4f} secondi")


