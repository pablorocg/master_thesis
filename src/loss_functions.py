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

    def __init__(self, classification_weight=1.0, margin=1.0):
        super(MultiTaskTripletLoss, self).__init__()
        self.margin = margin
        self.classification_weight = classification_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, 
                x_anchor, x_positive, x_negative, 
                class_anchor, class_positive, class_negative, 
                target_anchor, target_positive, target_negative):
        # Calcular las distancias de disimilitud
        D_p = F.pairwise_distance(x_anchor, x_positive, p=2)
        D_n = F.pairwise_distance(x_anchor, x_negative, p=2)

        # Calcular la pérdida
        margin_loss = torch.mean(F.relu(D_p - D_n + self.margin))
        
        # Calcular la pérdida de clasificación (Cross Entropy Loss)
        classification_loss_a = self.cross_entropy_loss(class_anchor, target_anchor)
        classification_loss_p = self.cross_entropy_loss(class_positive, target_positive)
        classification_loss_n = self.cross_entropy_loss(class_negative, target_negative)

        # Combinar las pérdidas
        loss = margin_loss + self.classification_weight * (classification_loss_a + classification_loss_p + classification_loss_n)
        return loss



# class InfoNCELoss(nn.Module):
#     """
#     InfoNCE Loss Function.

#     La función de pérdida se define como:

#         L_NCE = -sum_{i=1}^{N} log(exp(sim(x_i, y_i) / τ) / sum_{j=1}^{K} exp(sim(x_i, y_j) / τ))

#     donde:
#         sim(x, y) es la similitud (e.g., producto punto) entre x e y,
#         τ es la temperatura,
#         K es el número de muestras negativas más una muestra positiva.

#     Args:
#         temperature (float): El valor de la temperatura τ. Default: 0.07

#     Inputs:
#         x (Tensor): Embeddings de los ejemplos.
#         y (Tensor): Embeddings de los ejemplos positivos correspondientes.
#         negatives (Tensor): Embeddings de los ejemplos negativos.

#     Output:
#         loss (Tensor): El valor del InfoNCE loss.
#     """

#     def __init__(self, temperature=0.07):
#         super(InfoNCELoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, x, y, negatives):
#         # Concatenar las muestras positivas y negativas
#         positives = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
#         negatives = negatives.unsqueeze(1)
#         samples = torch.cat([positives, negatives], dim=1)

#         # Calcular las similitudes
#         sim = torch.matmul(x, samples.transpose(2, 1)) / self.temperature

#         # Crear las etiquetas (positivas en la posición 1)
#         labels = torch.zeros(sim.shape[0], dtype=torch.long, device=sim.device)

#         # Calcular la pérdida usando Cross Entropy
#         loss = F.cross_entropy(sim, labels)
#         return loss

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        """
        Args:
            features: tensor de forma (batch_size, feature_dim)
                      features de todos los grafos en el batch.
            labels: tensor de forma (batch_size,)
                    etiquetas de los grafos en el batch.
        """
        # Normalizar los features
        features = F.normalize(features, dim=1)

        batch_size = features.shape[0]

        # Crear la matriz de similaridad
        similarity_matrix = torch.matmul(features, features.T)

        # Aplicar temperatura
        similarity_matrix = similarity_matrix / self.temperature

        # Crear las etiquetas para la pérdida de contraste
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        # Obtener los targets
        targets = mask / mask.sum(1, keepdim=True)

        # Calcular la pérdida
        loss = -torch.sum(targets * F.log_softmax(similarity_matrix, dim=-1), dim=-1)
        return loss.mean()