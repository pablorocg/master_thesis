import torch
from tabulate import tabulate

class CFG:
    # Configuración general
    debug = True
    epochs = 1
    batch_size = 128
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-2
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    pos_neg_ratio = 1/3
    
    
    # Configuración del modelo de grafo
    graph_model_name = "GraphConvolutionalNetwork"#"GraphAttentionNetwork"
    graph_channels = 3
    graph_hidden_channels = 128
    graph_embedding = 512
    heads = 4
    dropout = 0.2
    n_hidden_blocks = 5
    graph_model_trainable = True

# tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    # Configuración del modelo de texto
    text_tokenizer = "medicalai/ClinicalBERT"
    text_encoder_model = "medicalai/ClinicalBERT"
    text_embedding = 768
    tex_model_trainable = True
    
    temperature = 1.0

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 512 
    dropout = 0.1

    n_classes = 32

    # Load model directly

    assert text_encoder_model in ["distilbert-base-uncased", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "medicalai/ClinicalBERT"]
    assert graph_model_name in ["GraphConvolutionalNetwork", "GraphAttentionNetwork"]
    
    
    print("Configuración del modelo:")
    print(tabulate([
        ["Modelo de grafo", graph_model_name],
        ["Canales de entrada", graph_channels],
        ["Canales ocultos", graph_hidden_channels],
        ["Embedding", graph_embedding],
        ["Cabezas de atención", heads],
        ["Dropout", dropout],
        ["Bloques ocultos", n_hidden_blocks],
        ["Modelo de grafo entrenable", graph_model_trainable],
        ["Modelo de texto", text_encoder_model],
        ["Embedding", text_embedding],
        ["Modelo de texto entrenable", tex_model_trainable],
        ["Temperatura", temperature],
        ["Capas de proyección", num_projection_layers],
        ["Dimensión de proyección", projection_dim],
        ["Dropout", dropout],
        ["Dispositivo", device],
        ["Tasa de aprendizaje", lr],
        ["Decaimiento de peso", weight_decay],
        ["Paciencia", patience],
        ["Factor", factor],
        ["Debug", debug]
    ], headers=["Parámetro", "Valor"]), end="\n\n")
