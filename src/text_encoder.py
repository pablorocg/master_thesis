import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from seaborn import heatmap




    # Corpus Callosum:
    #     Regions Connected: Connects the left and right cerebral hemispheres.
    #     Function: Facilitates communication between the hemispheres.
    #     Brain Region: Central, running between the hemispheres.
    #     Type of Tract: Commissural fibers.

    # Anterior Commissure:
    #     Regions Connected: Connects parts of the temporal lobes across the hemispheres.
    #     Function: Involved in pain, olfaction, and other functions.
    #     Brain Region: Central, anterior to the third ventricle.
    #     Type of Tract: Commissural fibers.

    # Internal Capsule:
    #     Regions Connected: Connects the cerebral cortex with the brainstem and spinal cord.
    #     Function: Carries motor and sensory information.
    #     Brain Region: Central, adjacent to the thalamus.
    #     Type of Tract: Projection fibers.

    # Corticospinal Tract:
    #     Regions Connected: Connects the cerebral cortex to the spinal cord.
    #     Function: Primarily involved in motor control.
    #     Brain Region: Descending tract through the brainstem to the spinal cord.
    #     Type of Tract: Projection fibers.

    # Arcuate Fasciculus:
    #     Regions Connected: Connects Broca's area and Wernicke's area.
    #     Function: Involved in language comprehension and production.
    #     Brain Region: Lateral, in the left (typically) cerebral hemisphere.
    #     Type of Tract: Association fibers.

    # Cingulum:
    #     Regions Connected: Connects parts of the limbic system including the cingulate gyrus and the entorhinal cortex.
    #     Function: Involved in emotion and memory.
    #     Brain Region: Encircles the corpus callosum.
    #     Type of Tract: Association fibers.

    # Superior Longitudinal Fasciculus:
    #     Regions Connected: Connects the frontal lobe to the posterior regions of the cerebral cortex.
    #     Function: Involved in attention, eye movement, and spatial awareness.
    #     Brain Region: Extends longitudinally along the lateral aspect of the hemisphere.
    #     Type of Tract: Association fibers.

    # Uncinate Fasciculus:
    #     Regions Connected: Connects the frontal lobe to the anterior temporal lobe.
    #     Function: Involved in memory and language.
    #     Brain Region: Lateral, in the anterior part of the brain.
    #     Type of Tract: Association fibers.

    # Optic Radiation:
    #     Regions Connected: Connects the lateral geniculate nucleus of the thalamus to the primary visual cortex.
    #     Function: Transmits visual information.
    #     Brain Region: Posterior, passing through the temporal and parietal lobes.
    #     Type of Tract: Projection fibers.

    # Fornix:
    #     Regions Connected: Connects the hippocampus to the hypothalamus.
    #     Function: Involved in memory processing.
    #     Brain Region: Central, part of the limbic system.
    #     Type of Tract: Projection fibers.

tracts = {
    "Corpus Callosum": {
        "Regions Connected": "Connects the left and right cerebral hemispheres.",
                "Function": "Facilitates communication between the hemispheres.",
                "Brain Region": "Central, running between the hemispheres.",
                "Type of Tract": "Commissural fibers."
            },"Anterior Commissure": {
                "Regions Connected": "Connects parts of the temporal lobes across the hemispheres.",
                "Function": "Involved in pain, olfaction, and other functions.",
                "Brain Region": "Central, anterior to the third ventricle.",
                "Type of Tract": "Commissural fibers."
            },"Internal Capsule": {
                "Regions Connected": "Connects the cerebral cortex with the brainstem and spinal cord.",
                "Function": "Carries motor and sensory information.",
                "Brain Region": "Central, adjacent to the thalamus.",
                "Type of Tract": "Projection fibers."
            },"Corticospinal Tract": {
                "Regions Connected": "Connects the cerebral cortex to the spinal cord.",
                "Function": "Primarily involved in motor control.",
                "Brain Region": "Descending tract through the brainstem to the spinal cord.",
                "Type of Tract": "Projection fibers."
            },"Arcuate Fasciculus": {
                "Regions Connected": "Connects Broca's area and Wernicke's area.",
                "Function": "Involved in language comprehension and production.",
                "Brain Region": "Lateral, in the left (typically) cerebral hemisphere.",
                "Type of Tract": "Association fibers."
            },"Cingulum": {
                "Regions Connected": "Connects parts of the limbic system including the cingulate gyrus and the entorhinal cortex.",
                "Function": "Involved in emotion and memory.",
                "Brain Region": "Encircles the corpus callosum.",
                "Type of Tract": "Association fibers."
            },"Superior Longitudinal Fasciculus": {
                "Regions Connected": "Connects the frontal lobe to the posterior regions of the cerebral cortex.",
                "Function": "Involved in attention, eye movement, and spatial awareness.",
                "Brain Region": "Extends longitudinally along the lateral aspect of the hemisphere.",
                "Type of Tract": "Association fibers."
            },"Uncinate Fasciculus": {
                "Regions Connected": "Connects the frontal lobe to the anterior temporal lobe.",
                "Function": "Involved in memory and language.",
                "Brain Region": "Lateral, in the anterior part of the brain.",
                "Type of Tract": "Association fibers."
            },"Optic Radiation": {
                "Regions Connected": "Connects the lateral geniculate nucleus of the thalamus to the primary visual cortex.",
                "Function": "Transmits visual information.",
                "Brain Region": "Posterior, passing through the temporal and parietal lobes.",
                "Type of Tract": "Projection fibers."
            },"Fornix": {
                "Regions Connected": "Connects the hippocampus to the hypothalamus.",
                "Function": "Involved in memory processing.",
                "Brain Region": "Central, part of the limbic system.",
                "Type of Tract": "Projection fibers."

            }
}


names = list(tracts.keys())
reg_connected = [tracts[name]["Regions Connected"] for name in names]
functions = [tracts[name]["Function"] for name in names]
brain_regions = [tracts[name]["Brain Region"] for name in names]
types = [tracts[name]["Type of Tract"] for name in names]


relacionar = [
    (names, names), # Nombre con nombre
    (names, reg_connected), # Nombre con regiones conectadas
    (names, functions), # Nombre con funciones
    (names, brain_regions), # Nombre con regiones del cerebro
    (names, types), # Nombre con tipos de tractos
]


for cat_1, cat_2 in relacionar:
    # Tokenize sentences
    inputs_1 = tokenizer(cat_1,
                            padding=True,
                            truncation=True,
                            return_tensors='pt')

    inputs_2 = tokenizer(cat_2,
                            padding=True,
                            truncation=True,
                            return_tensors='pt')



    # Compute token embeddings
    with torch.no_grad():
        embeddings_1 = model(**inputs_1
        embeddings_2 = model(**inputs_2)


    # Perform pooling. In this case, mean pooling.
    embeddings_1 = meanpooling(embeddings_1, inputs_1['attention_mask'])

    embeddings_2 = meanpooling(embeddings_2, inputs_2['attention_mask'])



    plot_cosine_similarity_matrix(embedding_1 = embeddings_1,
                                embedding_2 = embeddings_2,
                                embedding_1_labels = cat_1,
                                embedding_2_labels = cat_2)