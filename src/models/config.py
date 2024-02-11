import torch

class CFG:
    debug = False
    # image_path = image_path
    # captions_path = captions_path
    batch_size = 32
    num_workers = 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_name = 'resnet50'
    # image_embedding = 2048
    graph_model_name = "GraphConvolutionalNetwork"
    graph_embedding = 1024
    # 3 si solo coordenadas y 4 si coordenadas y valor en imagen T1w
    graph_channels = 4

    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 10

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # # image size
    # size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1