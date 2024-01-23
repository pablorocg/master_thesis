import pathlib2 as pathlib
import torch
from torch_geometric.data import Dataset, Data
import random

class GraphFiberDataset(Dataset):
    def __init__(self, root, n_files, m_data, transform=None, pre_transform=None):
        self.n_files = n_files
        self.m_data = m_data
        super(GraphFiberDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # No se necesitan archivos crudos ya que estamos cargando datos ya procesados.
        return []

    @property
    def processed_file_names(self):
        # Seleccionar n archivos aleatoriamente para el dataset.
        all_files = os.listdir(self.processed_dir)
        selected_files = random.sample(all_files, self.n_files)
        return selected_files

    def download(self):
        # No es necesario ya que los datos ya están procesados.
        pass

    def process(self):
        # No es necesario ya que los datos ya están procesados.
        pass

    def len(self):
        return self.n_files * self.m_data

    def get(self, idx):
        # Cargar un ejemplo de datos por índice de forma perezosa.
        file_idx = idx // self.m_data
        data_idx = idx % self.m_data
        
        # Carga el archivo seleccionado y extrae el objeto Data.
        file_path = os.path.join(self.processed_dir, self.processed_file_names[file_idx])
        all_data = torch.load(file_path)
        
        # Seleccionar m_data elementos Data de forma aleatoria.
        data = random.sample(all_data, self.m_data)[data_idx]
        
        # Extraer la etiqueta del nombre del archivo.
        label = self._extract_label(self.processed_file_names[file_idx])
        data.y = label
        
        return data
    
    def _extract_label(self, file_name):
        # Extrae la etiqueta del nombre del archivo.
        # Asumiendo que la etiqueta es parte del nombre del archivo.
        label_part = file_name.split('_')[1]  # Esta es una suposición, ajusta según tu esquema de nombres.
        label = int(label_part)  # Convierte a int o a la forma que necesites.
        return label

# Para usar el conjunto de datos:
root_dir = 'path/to/data'
n_files = 10  # Número de archivos a seleccionar.
m_data = 5    # Número de elementos Data a extraer de cada archivo.
dataset = LazyDataset(root=root_dir, n_files=n_files, m_data=m_data)



# Path: src/data_loader_graph.py
ds = GraphFiberDataset(root='tractoinferno_graphs')
print(ds)
