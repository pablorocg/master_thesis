import torch

class CFG:
    debug = True
    batch_size = 128
    num_workers = 8
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    graph_model_name = "GraphConvolutionalNetwork"#"GraphAttentionNetwork"
    graph_embedding = 1024
    # 3 si solo coordenadas y 4 si coordenadas y valor en imagen T1w
    graph_channels = 3

    # Posibles valores 
    text_encoder_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    text_embedding = 30522
    text_tokenizer = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    max_length = 10

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 768 
    dropout = 0.1

    # Load model directly
# 

# tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
# model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")