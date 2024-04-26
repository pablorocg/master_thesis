import pytorch_lightning as pl
import torchmetrics
from torch import optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules import TextGraphModule
from losses import CategoricalContrastiveLoss
from losses import CategoricalContrastiveLoss
import torchmetrics





config = {
    "theta": 0.5,
    "margin": 0.5,
    "distance": 'euclidean',
    "weighted_loss": False,
    "n_classes": 32,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_workers": 4,
    "text_encoder_name": "emilyalsentzer/Bio_ClinicalBERT",
    "text_embedding": 768,
    "graph_model_name": "GCN",
    "graph_embedding": 768,
    "graph_channels": 3,
    "projection_dim": 256,
}




class TextGraphModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Guardar hiperparámetros
        self.save_hyperparameters(config)

        # Inicializar modelo
        self.model = TextGraphModule(text_encoder_name = config['text_encoder_name'],
                                     text_embedding = config['text_embedding'],
                                     graph_model_name = config['graph_model_name'],
                                     graph_embedding = config['graph_embedding'],
                                     graph_channels = config['graph_channels'],
                                     projection_dim = config['projection_dim'],
                                     n_classes = config['n_classes'])
 

        # Inicializar funcion de pérdida
        self.criterion = CategoricalContrastiveLoss(theta=config['theta'], 
                                                    margin=config['margin'], 
                                                    dw=config['distance'])

        # Inicializar métricas
        # self.auroc_graphs = MulticlassAUROC(config['n_classes'], "macro")
        # self.auroc_texts = MulticlassAUROC(config['n_classes'], "macro")
        self.accuracy_graphs = torchmetrics.classification.Accuracy(task="multiclass", num_classes=config['n_classes'])
        self.accuracy_texts = torchmetrics.classification.Accuracy(task="multiclass", num_classes=config['n_classes'])
        
        # self.f1_graphs = MulticlassF1Score(config['n_classes'], "macro")
        # self.f1_texts = MulticlassF1Score(config['n_classes'], "macro")


        
    def forward(self, graph_data, text_data):
        return self.model(graph_data, text_data) # Devuelve los embeddings de los grafos y textos y las predicciones de las etiquetas (4 tensores)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                                lr=self.hparams.learning_rate, 
                                weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, patience=1500, factor=0.95, verbose=True)
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler, 
                "monitor": "train_loss"}
    
    def training_step(self, batch, batch_idx):
        # Implementar el paso de entrenamiento
        graph_data, text_data, graph_label, text_label, type_of_pair = batch
        g_proj, t_proj, g_pred_lab, t_pred_lab = self(graph_data, text_data)
        loss = self.criterion(g_proj, t_proj, graph_label, text_label, g_pred_lab, t_pred_lab, type_of_pair)
        
        self.accuracy_graphs(g_pred_lab, graph_label)
        self.accuracy_texts(t_pred_lab, text_label)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('graphs_train_acc', self.accuracy_graphs, prog_bar=True, on_step=True, on_epoch=False)
        self.log('texts_train_acc', self.accuracy_texts, prog_bar=True, on_step=True, on_epoch=False)

        
        return loss

    def validation_step(self, batch, batch_idx):
        # Implementar el paso de validación
        graph_data, text_data, graph_label, text_label, type_of_pair = batch
        g_proj, t_proj, g_pred_lab, t_pred_lab = self(graph_data, text_data)
        loss = self.criterion(g_proj, t_proj, graph_label, text_label, g_pred_lab, t_pred_lab, type_of_pair)
        self.accuracy_graphs(g_pred_lab, graph_label)
        self.accuracy_texts(t_pred_lab, text_label)

        self.log("valid_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('graphs_valid_acc', self.accuracy_graphs, prog_bar=True, on_step=True, on_epoch=False)
        self.log('texts_valid_acc', self.accuracy_texts, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Implementar el paso de test
        graph_data, text_data, graph_label, text_label, type_of_pair = batch
        g_proj, t_proj, g_pred_lab, t_pred_lab = self(graph_data, text_data)
        loss = self.criterion(g_proj, t_proj, graph_label, text_label, g_pred_lab, t_pred_lab, type_of_pair)
        # Hacer log del loss en el paso de test
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_training_epoch_end(self):
        # Implementar el final de la época de entrenamiento
        self.log("train_loss", self.trainer.callback_metrics['train_loss'])

