import torch
from tabulate import tabulate

class CFG:
    
    # Configuración general
    train_epochs = 1
    train_batch_size = 1024
    train_num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-3
    optimizer_weight_decay = 1e-4# 1e-3
    scheduler_patience = 1
    log = True
    

    # Configuracion para la funcion de collate en el DataLoader
    pos_neg_pair_ratio = 1/5 # 1 par positivo por cada 5 pares negativos
    
    
    # Configuración del modelo de grafo
    graph_encoder_name ="GraphConvolutionalNetwork" # Tipo de encoder de grafo que se usará
    graph_encoder_input_channels = 3 # Canales de entrada del grafo (número de características) (x, y, z)
    graph_encoder_hidden_channels = 128 # Canales ocultos del encoder de grafo
    graph_encoder_graph_embedding = 128 # Dimensión de la capa de salida del encoder de grafo
    graph_encoder_dropout = 0.2 # Dropout del encoder de grafo
    graph_encoder_n_hidden_blocks = 5 # Número de bloques ocultos del encoder de grafo
    
    
    text_encoder_tokenizer = "emilyalsentzer/Bio_ClinicalBERT"
    text_encoder_name = "emilyalsentzer/Bio_ClinicalBERT"
    text_encoder_embedding = 768
    text_encoder_trainable = False
    
    # for projection head; used for both image and text encoders
    projection_head_output_dim = 128 # 128
    projection_head_dropout = 0.15

    n_classes = 32

    # Contrastive loss parameters
    theta = 0.5
    margin = 1.0
    distance = 'cosine'
    weighted_loss = False
    assert distance in ['cosine', 'euclidean'], f"Distance {distance} not supported"


if __name__ == "__main__":
    print(tabulate([(name, value) for name, value in CFG.__dict__.items() if not name.startswith("__")], headers=("Name", "Value")))
    print(CFG.__dict__.items())

    # Crear un diccionario con los valores de la configuración
    config = {name: value for name, value in CFG.__dict__.items() if not name.startswith("__")}
    print(config)

    
 
