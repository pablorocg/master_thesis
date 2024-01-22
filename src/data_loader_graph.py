import pathlib2 as pathlib
import torch
from torch_geometric.data import Dataset, Data

class MyCustomDataset(Dataset):
    def __init__(self, 
                 root = r"C:\Users\pablo\GitHub\tfm_prg\tractoinferno_graphs", 
                 transform=None, 
                 pre_transform=None):
        super(MyCustomDataset, self).__init__(root, transform, pre_transform)
        # self.data_files = os.listdir(self.processed_dir)
        self.data_files = [file for file in pathlib.Path(self.processed_dir).rglob('*.pt')]

    @property
    def raw_file_names(self):
        # Retorna una lista de nombres de archivos crudos que necesitas procesar.
        return ['file1', 'file2', ...]

    @property
    def processed_file_names(self):
        # Retorna una lista de nombres de archivos procesados que se guardarán.
        return self.data_files

    

    def process(self):
        # Procesa los archivos crudos y guarda los datos en formato de Torch en self.processed_dir.
        for raw_path in self.raw_paths:
            # Procesa cada archivo crudo y guarda los datos.
            data = ... # Crea una instancia de Data aquí.
            torch.save(data, os.path.join(self.processed_dir, 'data_x.pt'))

    def len(self):
        # Retorna el número de ejemplos en tu conjunto de datos.
        return len(self.data_files)

    def get(self, idx):
        # Carga un ejemplo de datos por índice.
        data = torch.load(os.path.join(self.processed_dir, self.data_files[idx]))
        return data

# Para usar tu conjunto de datos:
dataset = MyCustomDataset(root='path/to/data')

dataset.processed_file_names

