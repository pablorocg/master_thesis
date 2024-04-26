import torch
from torch import nn
import torch.nn.functional as F


class CategoricalContrastiveLoss(nn.Module):
    def __init__(self, theta:float, margin:float, dw:str):
        super(CategoricalContrastiveLoss, self).__init__()
        self.theta = theta
        self.margin = margin  # margen
        self.classification_loss = nn.CrossEntropyLoss()
        self.dw = dw

    def forward(self, graph_emb, text_emb, graph_label, text_label, graph_pred_label, text_pred_label, y):
        # Calcula la pérdida de disimilitud como la distancia euclídea
        if self.dw == 'euclidean':
            graph_emb = F.normalize(graph_emb, p=2, dim=1)
            text_emb = F.normalize(text_emb, p=2, dim=1)
            dw = torch.norm(graph_emb - text_emb, p=2, dim=1)
        
        elif self.dw == 'cosine':
            graph_emb = F.normalize(graph_emb, p=2, dim=1)
            text_emb = F.normalize(text_emb, p=2, dim=1)
            dw = 1 - F.cosine_similarity(graph_emb, text_emb, dim=1)

        loss_similar = (1 - y) * torch.pow(dw, 2)# Pérdida para pares similares: Ls(Dw) = Dw^2
        loss_dissimilar = y * torch.pow(F.relu(self.margin - dw), 2)# Pérdida para pares disímiles: Ld(Dw) = max(0, m - Dw)^2
        contrastive_loss = loss_similar + loss_dissimilar
        
        graph_classification_loss = self.classification_loss(graph_pred_label, graph_label)
        text_classification_loss = self.classification_loss(text_pred_label, text_label)
        
        loss = contrastive_loss + self.theta * (graph_classification_loss + text_classification_loss)

        return loss.mean()