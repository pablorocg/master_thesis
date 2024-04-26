from transformers import AutoModel
import torch.nn as nn
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from torch.nn import ModuleList


class TextEncoder(nn.Module):
    """Text Encoder that wraps a pretrained transformer model for embedding extraction."""

    def __init__(self, 
                pretrained_model_name_or_path:str,
                trainable: bool = False) -> None:
        """
        Initializes the TextEncoder with a pretrained model.
        """
        super(TextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        
        # Freeze the model's parameters if not trainable
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0
        

    def forward(self, tokenized_text: dict) -> torch.Tensor:
        """
        Performs a forward pass through the model to obtain embeddings.
        """
        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']
        
        output = self.model(input_ids=input_ids, 
                            attention_mask=attention_mask)
        
        return output.last_hidden_state[:, self.target_token_idx, :]
        
        
    
    def meanpooling(self, 
                    output: torch.Tensor, 
                    mask: torch.Tensor) -> torch.Tensor:
        """
        Applies mean pooling on the model's output embeddings, taking the attention mask into account.
        """

        embeddings = output[0]  # The first element of model_output contains all token embeddings.
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()

        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        

class ProjectionHead(nn.Module):
    """
    Proyección de las embeddings de texto a un espacio de dimensión reducida.
    """
    def __init__(
        self,
        embedding_dim:int,# Salida del modelo de lenguaje (768)
        projection_dim:int, # Dimensión de la proyección (256)
        dropout:float
    ):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ClassifierHead(nn.Module):
    """
    Capa FC con activación softmax para que clasifique la clase.

    Parametros:
    - projection_dim: Dimensión de la proyección (512)
    - n_classes: Número de clases a clasificar (32)

    Salida:
    - x: LogSoftmax de las predicciones de las clases
    """
    def __init__(
        self,
        projection_dim:int, # Dimensión de la proyección (512)
        n_classes:int # Número de clases a clasificar (32)
    ):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(projection_dim, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)





class GraphConvBlock(nn.Module):
    """
    Bloque de capas de convolución de grafos.
    
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(GraphConvBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.LeakyReLU()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return x
    

class GCNEncoder(nn.Module):
    """
    Codificador de grafos que utiliza capas de convolución de grafos.

    """
    def __init__(self, in_channels, hidden_dim, out_channels, dropout, n_hidden_blocks):
        super(GCNEncoder, self).__init__()
        self.input_block = GraphConvBlock(in_channels, hidden_dim, dropout)
        self.hidden_blocks = nn.ModuleList(
            [
                GraphConvBlock(hidden_dim, hidden_dim, dropout) 
                for _ in range(n_hidden_blocks - 1)
            ]
        )
        self.output_block = GraphConvBlock(hidden_dim, out_channels, dropout)
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_block(x, edge_index)

        for layer in self.hidden_blocks:
            x = layer(x, edge_index)

        x = self.output_block(x, edge_index)

        return global_mean_pool(x, batch)



class TextGraphModule(nn.Module):
    def __init__(self, 
                 text_encoder_name, 
                 text_embedding, 
                 graph_model_name,
                 graph_embedding,
                 graph_channels,
                 projection_dim,
                 n_classes):
        
        super(TextGraphModule, self).__init__()
        if graph_model_name == "GCN":
            self.graph_encoder = GCNEncoder(in_channels=graph_channels, 
                                            hidden_dim=graph_embedding,
                                            out_channels=graph_embedding,
                                            dropout=0.3,
                                            n_hidden_blocks=3) #(batch_size, graph_embedding)
        
        self.graph_projection_head = ProjectionHead(embedding_dim=graph_embedding, 
                                                    projection_dim=projection_dim, 
                                                    dropout=0.3) #(batch_size, projection_dim)
        self.graph_embedding_classifier = ClassifierHead(projection_dim=projection_dim, 
                                                         n_classes=n_classes)


        self.text_encoder = TextEncoder(pretrained_model_name_or_path=text_encoder_name, 
                                        trainable=False) #(batch_size, text_embedding)
        self.text_projection_head = ProjectionHead(embedding_dim=text_embedding, 
                                                   projection_dim=projection_dim, 
                                                   dropout=0.3) #(batch_size, projection_dim)
        self.text_embedding_classifier = ClassifierHead(projection_dim=projection_dim, 
                                                        n_classes=n_classes)
        
        
    
    def forward(self, graph_batch, text_batch):
        
        graph_projections = self.graph_encoder(graph_batch) # (batch_size, projection_dim)
        graph_projections = self.graph_projection_head(graph_projections) # (batch_size, projection_dim)

        graph_predicted_labels = self.graph_embedding_classifier(graph_projections) # (batch_size, n_classes)
        
        text_projections = self.text_encoder(text_batch) # (batch_size, text_embedding)
        text_projections = self.text_projection_head(text_projections) # (batch_size, projection_dim)

        text_predicted_labels = self.text_embedding_classifier(text_projections) # (batch_size, n_classes)

        return graph_projections, text_projections, graph_predicted_labels, text_predicted_labels
    

if __name__ == '__main__':
    
    model = TextGraphModule(text_encoder_name = "emilyalsentzer/Bio_ClinicalBERT",
                            text_embedding = 768,
                            graph_model_name = "GCN",
                            graph_embedding = 768,
                            graph_channels = 3,
                            projection_dim = 256,
                            n_classes = 32)
    
    print(model)