import pathlib2 as pathlib
from nibabel import load
from dipy.io.streamline import load_trk
import numpy as np
import torch
from torch_geometric.data import Data
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
        
    
    def min_max_normalization(self, data:torch.tensor) -> torch.tensor:
        """
        Normaliza los valores de una imagen entre 0 y 1.
        """
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    
    
        
    def generate_graphs_from_subject(self, subject_dict:dict) -> Data:
        """
        Genera un grafo por cada fibra de una imagen T1w.
        """

        subject_id = subject_dict["subject"]      # str (subj-01, subj-02, etc)
        t1_img = subject_dict["T1w"]          # Path (ruta a la imagen T1w)
        tracts = subject_dict["tracts"]       # List[Path] (lista de rutas a las fibras del sujeto)
        split = subject_dict["subject_split"] # str (trainset, testset o validset)
        
        subj = load(str(t1_img))
        anat = torch.tensor(subj.get_fdata(), dtype=torch.float)
        norm_anat = self.min_max_normalization(anat)# Normalizar la imagen

        affine = torch.tensor(subj.affine, dtype=torch.float)
        inv_affine = torch.inverse(affine)# Obtener la inversa de la matriz de transformación con torch
        
        

        for tract in tracts:
            
            subject_graphs = []
            
            text_label = tract.stem.split("__")[1].split("_m")[0]# Obtener el label de la fibra
            label = ds_handler.get_label_from_tract(text_label)
            label = torch.tensor(label, dtype = torch.long)
            
            tractogram = load_trk(str(tract), 'same')
            streamlines, affine = tractogram.streamlines, tractogram.affine

            for streamline in streamlines:
                streamline_tensor = torch.from_numpy(streamline).float()
                xyz1 = torch.cat((streamline_tensor, torch.ones(streamline_tensor.shape[0], 1)), dim=1)
                ijk = torch.mm(inv_affine, xyz1.T).T[:, :3]
                vox_ids = ijk.round().long()
                # Extraer los valores de la imagen en los indices de la fibra
                values = norm_anat[vox_ids[:, 0], vox_ids[:, 1], vox_ids[:, 2]].unsqueeze(1)
                # Concatenar los valores de la imagen a la fibra
                nodes = torch.cat((streamline_tensor, values), dim=1)
                # Crear aristas no dirigidas
                edges = [[i, i+1] for i in range(len(nodes) - 1)] + [[i+1, 1] for i in range(len(nodes) - 1)]# Torch geometric necesita que las aristas [i, j] y [j, i] esten contiguas para que sea no dirigido

                graph = Data(x = nodes, 
                             edge_index = torch.tensor(edges).t().contiguous(),# Transponer y hacer contiguas las aristas para que sea no dirigido
                             y = label)
                # Añadir el grafo a la lista de grafos
                subject_graphs.append(graph)

            # Guardar los grafos en un archivo .pt
            # Si no existe el directorio, crearlo
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)
            # Si no existe el directorio de la partición, crearlo
            if not self.output_dir.joinpath(split).exists():
                self.output_dir.joinpath(split).mkdir(parents=True)
            torch.save(subject_graphs, self.output_dir.joinpath(split, f"{subject_id}__{text_label}.pt"))
        return subject_graphs 
    
    def generate_graphs_from_subjects(self, subjects_list:list) -> None:
        """
        Genera los grafos de una lista de sujetos.
        """
        for subject in tqdm(subjects_list):
            self.generate_graphs_from_subject(subject)



if __name__ == "__main__":
    ds_handler = Tractoinferno_handler(tractoinferno_path = r"C:\Users\pablo\GitHub\tfm_prg\tractoinferno_preprocessed_mni", scope="testset")
    graph_generator = Graph_generator(output_dir = r"C:\Users\pablo\GitHub\tfm_prg\tractoinferno_graphs", ds_handler = ds_handler)
    graph_generator.generate_graphs_from_subjects(ds_handler.get_data())
