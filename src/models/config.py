import torch
from tabulate import tabulate

class CFG:
    def __init__(self):
        # Configuración general
        self.train_epochs = 1
        self.train_batch_size = 64
        self.train_num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 1e-3
        self.optimizer_weight_decay = 1e-4# 1e-3
        self.scheduler_patience = 1
        

        # Configuracion para la funcion de collate en el DataLoader
        self.pos_neg_pair_ratio = 1/5 # 1 par positivo por cada 5 pares negativos
        
        
        # Configuración del modelo de grafo
        self.graph_encoder_name ="GraphConvolutionalNetwork" # Tipo de encoder de grafo que se usará
        self.graph_encoder_input_channels = 3 # Canales de entrada del grafo (número de características) (x, y, z)
        self.graph_encoder_hidden_channels = 128 # Canales ocultos del encoder de grafo
        self.graph_encoder_graph_embedding = 128 # Dimensión de la capa de salida del encoder de grafo
        self.graph_encoder_dropout = 0.2 # Dropout del encoder de grafo
        self.graph_encoder_n_hidden_blocks = 5 # Número de bloques ocultos del encoder de grafo
        
        
        self.text_encoder_tokenizer = "emilyalsentzer/Bio_ClinicalBERT"
        self.text_encoder_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.text_encoder_embedding = 768
        self.text_encoder_trainable = True
        
        # for projection head; used for both image and text encoders
        self.projection_head_output_dim = 128 # 128
        self.projection_head_dropout = 0.15

        self.n_classes = 32

        # Contrastive loss parameters
        self.theta = 0.5
        self.margin = 1.0
        self.distance = 'cosine'
        self.weighted_loss = True
        assert self.distance in ['cosine', 'euclidean'], f"Distance {self.distance} not supported"


if __name__ == "__main__":
    cfg = CFG()
    print(tabulate([(name, value) for name, value in vars(cfg).items()]))

    
 
