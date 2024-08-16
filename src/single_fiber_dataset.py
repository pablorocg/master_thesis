from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from dipy.io.streamline import load_trk
from dipy.tracking.streamline import select_random_set_of_streamlines

import torch
from utils import create_graph
from nibabel.streamlines import ArraySequence



class StreamlineSingleDataset(Dataset):
    def __init__(self, datadict: dict, ds_handler, transform=None, select_n_streamlines=1000):
        self.subject = datadict  # Subject data dictionary
        self.transform = transform  # Transform to apply to each graph
        self.streamlines = ArraySequence()
        self.labels = []

        # Cargar los tractos, eliminar streamlines inválidas y construir las estructuras en un solo bucle
        for tract in self.subject["tracts"]:
            tract_label = ds_handler.get_label_from_tract(tract.stem)

            # Cargar el tracto y remover las streamlines inválidas
            tractogram = load_trk(str(tract), 'same', bbox_valid_check=False)
            # recalculate the bounding box
            # tractogram.to_vox()
            # tractogram.to_rasmm()

            tractogram.remove_invalid_streamlines()
            streamlines = tractogram.streamlines
            # streamlines.remove_invalid_streamlines()
            if select_n_streamlines is not None and select_n_streamlines >= 0:
                
                streamlines = select_random_set_of_streamlines(streamlines, select_n_streamlines)

            # Añadir las streamlines al ArraySequence y las etiquetas como tensor
            self.streamlines.extend(streamlines)
            self.labels.extend([tract_label] * len(streamlines))

        # Convertir self.labels a un tensor de PyTorch
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.streamlines)

    def __getitem__(self, idx):
        streamline = self.streamlines[idx]
        label = self.labels[idx]
        graph = create_graph(streamline, label)
        if self.transform:
            graph = self.transform(graph)
        return graph

def collate_single_ds(data_list):
    return Batch.from_data_list(data_list)