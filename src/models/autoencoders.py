import torch_geometric.nn as gnn
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import dropout_adj


class GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(GCN_Block, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x
    

class GCN_Encoder(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 hidden_dim=64, 
                 out_channels=768,  # Asume que out_channels es la dimensión del espacio latente para mu y logstd
                 dropout=0.15):
        super(GCN_Encoder, self).__init__()
        
        # First block
        self.first_block = GCN_Block(in_channels, hidden_dim, dropout)
        
        # Hidden blocks
        self.hidden_blocks = nn.ModuleList([GCN_Block(hidden_dim, hidden_dim, dropout) for _ in range(2)])
        
        # Bottleneck
        self.mu = GCNConv(hidden_dim, out_channels)
        self.logstd = GCNConv(hidden_dim, out_channels)


        
    def forward(self, x, edge_index):
        x = self.first_block(x, edge_index)
        for block in self.hidden_blocks:
            x = block(x, edge_index)
        mu = self.mu(x, edge_index)
        logstd = self.logstd(x, edge_index)
        return mu, logstd
    



    


if __name__ == "__main__":
    from torch_dataset import FiberGraphDataset, collate_function
    from torch_geometric.data import Batch as GeoBatch
    from tqdm import tqdm
    # import dataloader
    from torch.utils.data import DataLoader
    import tensorboard
    import time
    import neptune
    import csv
    import numpy as np
    from torch_geometric.nn import global_mean_pool
    # import SummaryWriter
    from torch.utils.tensorboard import SummaryWriter


    ds = FiberGraphDataset(root='/app/dataset/tractoinferno_graphs/testset')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")


    model = VGAE(GCN_Encoder())
    model = model.to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=500, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08, verbose=False)

    log = True
    if log:
        name = f"model_{time.time()}"
            
        run = neptune.init_run(
            project="pablorocamora/multimodal-fiber-classification",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODA0YzA2NS04MjczLTQyNzItOGE5Mi05ZmI5YjZkMmY3MDcifQ==",
            name=name
        )

    params = {
        'epochs': 10,
        'batch_size': 128,
        'num_workers': 6,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'patience': 500,
        'factor': 0.99
    }

    for epoch in range(params['epochs']):
        
        for idx_suj, suj in enumerate(ds):
            if idx_suj == 3:
                break
            # progress bar with tqdm
            print(f"Processing subject {idx_suj+1}/{len(ds)}")
            dataloader = DataLoader(suj, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'],  collate_fn=GeoBatch.from_data_list)
            progress_bar = tqdm(dataloader, total=len(dataloader))

            for batch_idx, batch in enumerate(progress_bar):
                batch = batch.to(device)
                optimizer.zero_grad()

                # Forward pass
                z = model.encode(batch.x, batch.edge_index)

                # Compute the loss
                loss = model.recon_loss(z, batch.edge_index)
                loss = loss + (1 / batch.num_graphs) * model.kl_loss(z)

                if log:
                    run["loss"].log(loss.item())

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step(loss)


                progress_bar.set_description(f"Loss: {loss.item()}")
                progress_bar.update(1)

            progress_bar.close()
                
                
            
            
           
        if log:
            # importar summary writer
            writer = SummaryWriter()


            # Carga el dataset específico y prepara el DataLoader
            dataloader = DataLoader(ds[10], batch_size=128, shuffle=True, num_workers=1, collate_fn=GeoBatch.from_data_list)

            # Inicializa listas para acumular embeddings y etiquetas
            all_embeddings = torch.Tensor()  # Inicializa un tensor vacío para acumular embeddings
            all_labels = []  # Inicializa una lista para acumular etiquetas

            model.eval()  # Pon tu modelo en modo evaluación

            progress_bar = tqdm(dataloader, total=len(dataloader), desc="Embedding")
            
            for idx, batch in enumerate(progress_bar):
                batch = batch.to(device)
                embeddings = model.encode(batch.x, batch.edge_index)  # Obtiene los embeddings
                embeddings = global_mean_pool(embeddings, batch.batch)  # Aplica Global Mean Pooling si es necesario
                
                # Acumula embeddings y etiquetas
                all_embeddings = torch.cat((all_embeddings, embeddings.detach().cpu()), dim=0)
                all_labels.extend(batch.y.cpu().numpy())# Convertir a etiquetas de texto si es necesario
                
                if idx == 100:
                    break

            

            # Convierte las etiquetas a un tensor para compatibilidad con add_embedding
            all_labels_tensor = torch.tensor(all_labels)

            # Guarda los embeddings y etiquetas en TensorBoard
            writer.add_embedding(all_embeddings, metadata=all_labels_tensor, global_step=0)
            writer.close()


                        
            
        



                
                    


    print("Done!")

            

