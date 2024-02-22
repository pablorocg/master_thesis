import pathlib2 as pathlib
from nibabel import load
from dipy.io.streamline import load_trk
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from dataset_handlers import Tractoinferno_handler
from tqdm import tqdm

class Graph_generator:
    def __init__(self, 
                 output_dir:str, 
                 ds_handler):
        # Comprobar si el directorio existe y si no, lanzar una excepción
        if not pathlib.Path(output_dir).exists():
            raise Exception(f"El directorio {output_dir} no existe.")
        else:
            self.output_dir = pathlib.Path(output_dir)

        self.ds_handler = ds_handler
        
    
    # def min_max_normalization(self, data:torch.tensor) -> torch.tensor:
    #     """
    #     Normaliza los valores de una imagen entre 0 y 1.
    #     """
    #     return (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    
    
        
    def generate_graphs_from_subject(self, subject_dict:dict) -> Data:
        """
        Genera un grafo por cada fibra de una imagen T1w.
        """

        subject_id = subject_dict["subject"]      # str (subj-01, subj-02, etc)
        tracts = subject_dict["tracts"]       # List[Path] (lista de rutas a las fibras del sujeto)
        split = subject_dict["subject_split"] # str (trainset, testset o validset)
        
        subject_graphs = []
        for tract in tracts:
            text_label = tract.stem.split("__")[1].split("_m")[0]# Obtener el label de la fibra
            label = ds_handler.get_label_from_tract(text_label)
            label = torch.tensor(label, dtype = torch.long)
            
            tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
            streamlines, affine = tractogram.streamlines, tractogram.affine

            for streamline in streamlines:
                nodes = torch.from_numpy(streamline).float()
                edges = torch.tensor([[i, i+1] for i in range(nodes.size(0)-1)] + [[i+1, i] for i in range(nodes.size(0)-1)], dtype=torch.long).T
                graph = Data(x = nodes, 
                             edge_index = edges,
                             y = label)
                # Almacenar grafo (Data) en estructura de datos
                subject_graphs.append(graph)
        
        # Convertir la lista de grafos en un solo objeto Data
        subject_graphs = Batch.from_data_list(subject_graphs)

        # Guardar los grafos en un archivo .pt
        self.output_dir.mkdir(parents=True, exist_ok=True)# Si no existe el directorio de la partición, crearlo
        output_path = str(self.output_dir.joinpath(split, f"{subject_id}.pt"))
        torch.save(subject_graphs, output_path)
        
        return subject_graphs    

    def generate_graphs_from_subjects(self, subjects_list:list) -> None:
        """
        Genera los grafos de una lista de sujetos.
        """
        for subject in tqdm(subjects_list):
            self.generate_graphs_from_subject(subject)



if __name__ == "__main__":
    ds_handler = Tractoinferno_handler(tractoinferno_path = "dataset/tractoinferno_preprocessed_mni", scope="testset")
    print(ds_handler.get_data())

    graph_generator = Graph_generator(output_dir = "dataset/tractoinferno_graphs", ds_handler = ds_handler)
    graph_generator.generate_graphs_from_subjects(ds_handler.get_data())


