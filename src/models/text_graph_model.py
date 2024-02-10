from encoders import GCN_Encoder, GAT_Encoder, CFG, Text_Encoder, ProjectionHead
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_geometric
from transformers import AutoTokenizer

class Multimodal_Text_Graph_Model(nn.Module):
    def __init__(self, 
                 text_encoder_model = CFG.text_encoder_model,
                 text_embedding = CFG.text_embedding,
                 graph_model_name = CFG.graph_model_name,
                 graph_embedding = CFG.graph_embedding,
                 graph_channels = CFG.graph_channels,
                 projection_dim = CFG.projection_dim,
                 temperature = CFG.temperature,
                 device = CFG.device):
        
        super(Multimodal_Text_Graph_Model, self).__init__()
        
        self.text_encoder = Text_Encoder(text_encoder_model)#, text_embedding)
        
        if graph_model_name == "GraphConvolutionalNetwork":
            self.graph_encoder = GCN_Encoder(graph_channels, graph_embedding)
        elif graph_model_name == "GraphAttentionNetwork":
            self.graph_encoder = GAT_Encoder(graph_channels, graph_embedding)
        else:
            raise ValueError("Invalid graph model name")
        
        # Proyectar representaciones a un espacio comun de menor dimension
        self.text_projection_head = ProjectionHead(text_embedding, projection_dim) 
        self.graph_projection_head = ProjectionHead(graph_embedding, projection_dim)
        self.temperature = temperature
        

    
    def forward(self, graph, text):
        # Encode the text and project the representations
        text_projections = self.text_encoder(text) # (batch_size, text_embedding)
        graph_projections = self.graph_encoder(graph) # (batch_size, graph_embedding)
        
        text_projections = self.text_projection_head(text_projections) # (batch_size, projection_dim)
        graph_projections = self.graph_projection_head(graph_projections) # (batch_size, projection_dim)
        
        # Calculating the Loss
        logits = (text_projections @ graph_projections.T) / self.temperature
        graphs_similarity = graph_projections @ graph_projections.T
        texts_similarity = text_projections @ text_projections.T
        targets = F.softmax(
            (graphs_similarity + texts_similarity) / 2 * self.temperature, dim = -1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        graphs_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (graphs_loss + texts_loss) / 2.0 # shape: (batch_size)
        
        return loss.mean()
    

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    

if __name__ == "__main__":
    import random
    import numpy as np
    import torch
    from torch_geometric.data import Data, Dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from torch_geometric.data import Batch as GeoBatch
    # Dataloader para el conjunto de datos de texto y grafos
    

    class RandomTextGraphDataset(Dataset):
        def __init__(self, root="", transform=None, pre_transform=None, num_samples=4000, num_max_nodes=30, num_node_features=4, return_tokenized_text=False):
            # Note: 'root' is required by PyG Dataset, but you might not need it for generating random data.
            super(RandomTextGraphDataset, self).__init__(root, transform, pre_transform)
            self.num_samples = num_samples # Numero de ejemplos en el dataset
            self.num_nodes = num_max_nodes # Numero maximo de nodos en cada grafo
            self.num_node_feat = num_node_features # Numero de caracteristicas de cada nodo
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.return_tokenized_text = return_tokenized_text

        def len(self):
            return self.num_samples

        def get(self, idx):
            graph_data = self.generate_random_graph_data()
            text = self.generate_random_text()
            if self.return_tokenized_text:
                return {'graph': graph_data, 'text': self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")}
            else:
                return {'graph': graph_data, 'text': text}

        def generate_random_text(self):
            words = ["left arcuate fasciculus", "right arcuate fasciculus", "left cingulum", "right fornix", "left fornix", "right cingulum", "left uncinate fasciculus", "right uncinate fasciculus"]
            return random.choice(words) # Seleccionar una palabra aleatoria

        def generate_random_graph_data(self):
            n = np.random.randint(20, self.num_nodes)
            x = torch.randn(n, self.num_node_feat)# (nun_nodes, num_node_features)
            edge_index = torch.tensor([[i, i+1] for i in range(n-1)] + [[i+1, i] for i in range(n-1)], dtype=torch.long).T # (2, num_edges)
            return Data(x=x, edge_index=edge_index) # Crear un objeto Data de PyG
    
    


    dataset = RandomTextGraphDataset(num_samples=40000, return_tokenized_text=True)
    
    

    def custom_collate_fn(batch):
        graphs = [item['graph'] for item in batch]
        input_ids = [item['text']['input_ids'].squeeze(0) for item in batch]  # Remueve la dimensión de batch innecesaria.
        attention_masks = [item['text']['attention_mask'].squeeze(0) for item in batch]  # Lo mismo para las máscaras de atención.

        # Aplica padding al nivel de batch. Esto garantiza que todos los tensores tengan la misma longitud.
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        batched_graphs = GeoBatch.from_data_list(graphs)

        # Puedes retornar también las máscaras de atención si tu modelo las necesita.
        return batched_graphs, {'input_ids': padded_input_ids, 'attention_mask': padded_attention_masks}

    # Crear el DataLoader con la función de collate_fn personalizada
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=custom_collate_fn)
    
    
    model = Multimodal_Text_Graph_Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    

    

    #Entrenar el modelo utilizando la gpu y tensorboard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    device = CFG.device
    print(f"Using device: {device}")
    model.to(device)
    for epoch in range(100):
        for i, (graph_data, text_data) in enumerate(dataloader):
            graph_data = graph_data.to(device)
            text_data = {key: val.to(device) for key, val in text_data.items()}
            loss = model(graph_data, text_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss, epoch * len(dataloader) + i)
            print(f"\r Epoch {epoch}, Iteration {i}, Loss {loss}")
    writer.flush()
    writer.close()

    
    
        
        

    
    
