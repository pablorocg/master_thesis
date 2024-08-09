import torch
import torch.nn as nn
import torch.nn.functional as F



class ContrastiveLoss(nn.Module):
    """
    Implementación de la pérdida contrastiva como se describe en la imagen proporcionada.
    
    La pérdida contrastiva se define como:
    
    L(w, (y, I_a, I_b)) = (1 - y) * L_S(D_w) + y * L_D(D_w)
    
    donde:
    - y ∈ {0, 1} es un indicador binario que representa si el par de entradas (I_a, I_b) es similar (y = 0) o disimilar (y = 1).
    - L_S(·) es la pérdida para pares similares.
    - L_D(·) es la pérdida para pares disimilares.
    - D_w es la distancia (en este caso, la distancia euclidiana) entre las salidas de la red para las entradas I_a e I_b.
    - La pérdida L_D(·) se define con un margen m tal que L_D = max(m - D_w, 0)^2.
    - La pérdida total para un conjunto P de pares de entrada se define como la suma de las pérdidas para cada par en P.

    L_S(w) = Σ_{i=1}^{|P|} L(w, (y, I_a, I_b)^(i))
    
    Args:
        margin (float): El margen para la pérdida de disimilaridad. Default: 1.0.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calcula la distancia euclidiana
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Calcula la pérdida de similaridad (LS) y disimilaridad (LD)
        LS = torch.mean((1 - label) * torch.pow(euclidean_distance, 2))
        LD = torch.mean(label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # Combina LS y LD para obtener la pérdida final
        loss = LS + LD
        return loss



class MultiTaskSiameseLoss(nn.Module):
    """
    Multi-Task Siamese Loss Function.

    La función de pérdida se define como:

        L_SC(w) = sum_{i=1}^{|P|} L(w, (y, I_a, I_b)^(i)) + θ [ L_C(ĉ_a, ζ(I_a))^(i) + L_C(ĉ_b, ζ(I_b))^(i) ]

    donde:
        L(w, (y, I_a, I_b)) es la pérdida de disimilitud,
        L_C es la pérdida de clasificación,
        D_w es la puntuación de similitud,
        θ es un peso para la contribución de la pérdida de clasificación.

    Args:
        classification_weight (float): El peso θ para la pérdida de clasificación. Default: 1.0

    Inputs:
        x_a (Tensor): Embeddings del ejemplo A.
        x_b (Tensor): Embeddings del ejemplo B.
        y (Tensor): Etiquetas indicando si los ejemplos son similares o no.
        class_a (Tensor): Clases predichas del ejemplo A.
        class_b (Tensor): Clases predichas del ejemplo B.
        target_a (Tensor): Clases reales del ejemplo A.
        target_b (Tensor): Clases reales del ejemplo B.

    Output:
        loss (Tensor): El valor del multi-task siamese loss.
    """

    def __init__(self, classification_weight=1.0):
        super(MultiTaskSiameseLoss, self).__init__()
        self.classification_weight = classification_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x_a, x_b, y, class_a, class_b, target_a, target_b):
        # Calcular la disimilitud (distancia euclidiana)
        D_w = F.pairwise_distance(x_a, x_b, p=2)

        # Calcular la pérdida de disimilitud (loss de margen contrastivo)
        margin_loss = torch.mean((1 - y) * torch.pow(D_w, 2) + 
                                 y * torch.pow(torch.clamp(1 - D_w, min=0.0), 2))

        # Calcular la pérdida de clasificación (Cross Entropy Loss)
        classification_loss_a = self.cross_entropy_loss(class_a, target_a)
        classification_loss_b = self.cross_entropy_loss(class_b, target_b)

        # Combinar las pérdidas
        loss = margin_loss + self.classification_weight * (classification_loss_a + classification_loss_b)
        return loss



class TripletLoss(nn.Module):
    """
    Triplet Loss Function.

    La función de pérdida se define como:

        L_T(w) = sum_{i=1}^{|J|} max(0, D_p - D_n + m)^(i)

    donde:
        D_p = d(x_r, x_p) y D_n = d(x_r, x_n) denotan las distancias del i-ésimo grupo, respectivamente,
        y m representa el margen de separación.

    Args:
        margin (float): El margen de separación para el triplet loss. Default: 1.0

    Inputs:
        x_anchor (Tensor): Embeddings del anchor.
        x_positive (Tensor): Embeddings del ejemplo similar.
        x_negative (Tensor): Embeddings del ejemplo disimilar.

    Output:
        loss (Tensor): El valor del triplet loss.
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x_anchor, x_positive, x_negative):
        # Calcular las distancias de disimilitud
        D_p = F.pairwise_distance(x_anchor, x_positive, p=2)
        D_n = F.pairwise_distance(x_anchor, x_negative, p=2)

        # Calcular la pérdida
        loss = torch.mean(F.relu(D_p - D_n + self.margin))
        return loss



class MultiTaskTripletLoss(nn.Module):
    """
    Multi-Task Triplet Loss Function.

    La función de pérdida se define como:

        L_SC(w) = sum_{i=1}^{|P|} L(w, (y, I_a, I_b)^(i)) + θ [ L_C(ĉ_a, ζ(I_a))^(i) + L_C(ĉ_b, ζ(I_b))^(i) ]

    donde:
        L(w, (y, I_a, I_b)) es la pérdida de disimilitud,
        L_C es la pérdida de clasificación,
        D_w es la puntuación de similitud,
        θ es un peso para la contribución de la pérdida de clasificación.

    Args:
        classification_weight (float): El peso θ para la pérdida de clasificación. Default: 1.0

    Inputs:
        x_a (Tensor): Embeddings del ejemplo A.
        x_b (Tensor): Embeddings del ejemplo B.
        y (Tensor): Etiquetas indicando si los ejemplos son similares o no.
        class_a (Tensor): Clases predichas del ejemplo A.
        class_b (Tensor): Clases predichas del ejemplo B.
        target_a (Tensor): Clases reales del ejemplo A.
        target_b (Tensor): Clases reales del ejemplo B.

    Output:
        loss (Tensor): El valor del multi-task siamese loss.
    """

    def __init__(self, classification_weight=1.0, margin=1.0, log=True, cross_entropy_weight_list = None):
        super(MultiTaskTripletLoss, self).__init__()
        self.margin = margin
        self.classification_weight = classification_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss(None if cross_entropy_weight_list is None else torch.tensor(cross_entropy_weight_list))
        self.log_dict = {}
        self.log = log

    def forward(self, 
                x_anchor, x_positive, x_negative, 
                class_anchor, class_positive,  class_negative, 
                target_anchor, target_positive, target_negative):
        
        # Verificar dimensiones de los embeddings
        assert x_anchor.size() == x_positive.size() == x_negative.size(), "Los embeddings deben tener las mismas dimensiones"
        assert class_anchor.size() == class_positive.size() == class_negative.size(), "Las clases deben tener las mismas dimensiones"
        
        # Calcular las distancias de disimilitud
        D_p = F.pairwise_distance(x_anchor, x_positive, p=2)
        D_n = F.pairwise_distance(x_anchor, x_negative, p=2)

        # Calcular la pérdida
        margin_loss = torch.mean(F.relu(D_p - D_n + self.margin))
        
        # Calcular la pérdida de clasificación (Cross Entropy Loss)
        classification_loss_a = self.cross_entropy_loss(class_anchor, target_anchor)
        classification_loss_p = self.cross_entropy_loss(class_positive, target_positive)
        classification_loss_n = self.cross_entropy_loss(class_negative, target_negative)
        class_avg_loss = (classification_loss_a + classification_loss_p + classification_loss_n) / 3

        if self.log:
            self.log_dict = {
                "margin_loss": margin_loss,
                "class_avg_loss": class_avg_loss,
                "weighted_class_loss": self.classification_weight * class_avg_loss
            }
            
        # Combinar las pérdidas
        loss = margin_loss + self.classification_weight * class_avg_loss
        return loss

#=========================================================================
import torch
import torch.nn.functional as F
from torch import nn




class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]