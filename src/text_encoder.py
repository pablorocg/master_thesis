import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from seaborn import heatmap

labels = ["casa", "perro", "gato", "casa", "perro", "gato"]
embeddings = torch.randn(6, 724)

# Calcula la matriz de similitud coseno entre todos los pares de vectores.
# Para esto, primero normaliza los vectores.
embeddings_norm = F.normalize(embeddings, p=2, dim=1)
similarities = torch.mm(embeddings_norm, embeddings_norm.T)

similarities = similarities.numpy()

mask = np.triu(np.ones_like(similarities))

similarities = mask * similarities

for row in similarities:
    position = np.argwhere(row == np.amax(row))
    print(position)
    

# Grafica la matriz de similitud con las labels utilizando heatmap.
plt.figure(figsize=(10, 10))
plt.imshow(similarities, cmap='hot')
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(np.arange(len(labels)), labels)
# El maximo de cada fila se coloca a rojo
plt.clim(0, 1)


plt.colorbar()

for row in similarities:
    print(row)

# sns.set(font_scale=1.2)
# heatmap(similarities, annot=True, xticklabels=labels, yticklabels=labels)
plt.show()




