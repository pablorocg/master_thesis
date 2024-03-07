# Importar librerías
import os
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import BaseTransform, Compose
from config import CFG
from transformers import AutoTokenizer
import random
from torch_geometric.data import Batch as GeoBatch


#=============================== Transformaciones preprocesado =================================
class MaxMinNormalization(BaseTransform):
        def __init__(self, max_values=None, min_values=None):
            """
            Initialize the normalization transform with optional max and min values.
            If not provided, they should be computed from the dataset.
            """
            self.max_values = max_values if max_values is not None else torch.tensor([76.03170776367188, 
                                                                                      77.9359130859375, 
                                                                                      88.72427368164062], 
                                                                                      dtype=torch.float)
            self.min_values = min_values if min_values is not None else torch.tensor([-73.90082550048828, 
                                                                                      -112.23554992675781, 
                                                                                      -79.38320922851562], 
                                                                                      dtype=torch.float)

        def __call__(self, data: Data) -> Data:
            """
            Apply min-max normalization to the node features.
            """
            data.x = (data.x - self.min_values) / (self.max_values - self.min_values)
            return data


#=============================== Clase para el dataset =================================
class FiberGraphDataset(Dataset):
    def __init__(self, 
                 root, 
                 transform = Compose([MaxMinNormalization()]), 
                 pre_transform = None):
        super(FiberGraphDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def processed_dir(self):
        return os.path.join(self.root)

    @property
    def processed_file_names(self):
        return os.listdir(self.root)
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        subject = self.processed_file_names[idx]# Seleccionar un sujeto
        graphs = torch.load(os.path.join(self.processed_dir, subject))
        
        if self.transform:
            graphs = self.transform(graphs)
        return graphs
    
# =============================== Clase para el batch de grafos ===============================
def collate_function(batch):
    """Funcion para el DataLoader"""
    tokenizer = AutoTokenizer.from_pretrained(CFG.graph_encoder_name, model_max_length=128, use_fast=True)#"bert-base-uncased"

    TRACT_LIST = {
        'AF_L': {'id': 0, 'tract': 'arcuate fasciculus', 'side' : 'left', 'type': 'association'},
        'AF_R': {'id': 1, 'tract': 'arcuate fasciculus','side' : 'right', 'type': 'association'},
        'CC_Fr_1': {'id': 2, 'tract': 'corpus callosum, frontal lobe', 'side' : 'most anterior part of the frontal lobe', 'type': 'commissural'},
        'CC_Fr_2': {'id': 3, 'tract': 'corpus callosum, frontal lobe', 'side' : 'most posterior part of the frontal lobe','type': 'commissural'},
        'CC_Oc': {'id': 4, 'tract': 'corpus callosum, occipital lobe', 'side' : 'central', 'type': 'commissural'},
        'CC_Pa': {'id': 5, 'tract': 'corpus callosum, parietal lobe', 'side' : 'central', 'type': 'commissural'},
        'CC_Pr_Po': {'id': 6, 'tract': 'corpus callosum, pre/post central gyri', 'side' : 'central', 'type': 'commissural'},
        'CG_L': {'id': 7, 'tract': 'cingulum', 'side' : 'left', 'type': 'association'},
        'CG_R': {'id': 8, 'tract': 'cingulum', 'side' : 'right', 'type': 'association'},
        'FAT_L': {'id': 9, 'tract': 'frontal aslant tract', 'side' : 'left', 'type': 'association'},
        'FAT_R': {'id': 10, 'tract': 'frontal aslant tract', 'side' : 'right', 'type': 'association'},
        'FPT_L': {'id': 11, 'tract': 'fronto-pontine tract', 'side' : 'left', 'type': 'association'},
        'FPT_R': {'id': 12, 'tract': 'fronto-pontine tract', 'side' : 'right', 'type': 'association'},
        'FX_L': {'id': 13, 'tract': 'fornix', 'side' : 'left', 'type': 'commissural'},
        'FX_R': {'id': 14, 'tract': 'fornix', 'side' : 'right', 'type': 'commissural'},
        'IFOF_L': {'id': 15, 'tract': 'inferior fronto-occipital fasciculus', 'side' : 'left', 'type': 'association'},
        'IFOF_R': {'id': 16, 'tract': 'inferior fronto-occipital fasciculus', 'side' : 'right', 'type': 'association'},
        'ILF_L': {'id': 17, 'tract': 'inferior longitudinal fasciculus', 'side' : 'left', 'type': 'association'},
        'ILF_R': {'id': 18, 'tract': 'inferior longitudinal fasciculus', 'side' : 'right', 'type': 'association'},
        'MCP': {'id': 19, 'tract': 'middle cerebellar peduncle', 'side' : 'central', 'type': 'commissural'},
        'MdLF_L': {'id': 20, 'tract': 'middle longitudinal fasciculus', 'side' : 'left', 'type': 'association'},
        'MdLF_R': {'id': 21, 'tract': 'middle longitudinal fasciculus', 'side' : 'right', 'type': 'association'},
        'OR_ML_L': {'id': 22, 'tract': 'optic radiation, Meyer loop', 'side' : 'left', 'type': 'projection'},
        'OR_ML_R': {'id': 23, 'tract': 'optic radiation, Meyer loop', 'side' : 'right', 'type': 'projection'},
        'POPT_L': {'id': 24, 'tract': 'pontine crossing tract', 'side' : 'left', 'type': 'commissural'},
        'POPT_R': {'id': 25, 'tract': 'pontine crossing tract', 'side' : 'right', 'type': 'commissural'},
        'PYT_L': {'id': 26, 'tract': 'pyramidal tract', 'side' : 'left', 'type': 'projection'},
        'PYT_R': {'id': 27, 'tract': 'pyramidal tract', 'side' : 'right', 'type': 'projection'},
        'SLF_L': {'id': 28, 'tract': 'superior longitudinal fasciculus', 'side' : 'left', 'type': 'association'},
        'SLF_R': {'id': 29, 'tract': 'superior longitudinal fasciculus', 'side' : 'right', 'type': 'association'},
        'UF_L': {'id': 30, 'tract': 'uncinate fasciculus', 'side' : 'left', 'type': 'association'},
        'UF_R': {'id': 31, 'tract': 'uncinate fasciculus', 'side' : 'right', 'type': 'association'}
    }

    LABELS = {value["id"]: key for key, value in TRACT_LIST.items()}# Diccionario id -> Etiqueta
    caption_templates = [
            "A {type} fiber",
            "A {type} fiber on the {side} side",
            "{type} fiber on the {side} side",
            "A {type} fiber of the {tract}",
            "{type} fiber of the {tract}",
            "A {type} fiber of the {tract} on the {side} side",
            "{type} fiber of the {tract} on the {side} side",
            "{side} side",
            "{tract} tract",
            "{type} fiber",
            "The {type} fiber located in the {tract} tract",
            "This is a {type} fiber found on the {side} hemisphere",
            "Detailed view of a {type} fiber within the {tract}",
            "Observation of the {type} fiber, prominently on the {side} side",
            "The {tract} tract's remarkable {type} fiber",
            "Characteristics of a {type} fiber in the {tract} region",
            "Notable {type} fiber on the {side} hemisphere of the {tract}",
            "Insight into the {type} fiber's structure on the {side} side",
            "Exploring the complexity of the {type} fiber in the {tract}",
            "The anatomy of a {type} fiber on the {side} hemisphere",
            "The {tract} tract featuring a {type} fiber",
            "A comprehensive look at the {type} fiber, {side} orientation",
            "A closer look at the {type} fiber's path in the {tract}",
            "Unveiling the {type} fiber's role in the {tract} tract",
            "Decoding the structure of the {type} fiber on the {side}",
            "Highlighting the {type} fiber's significance in the {tract}",
            "The {type} fiber: A journey through the {tract} on the {side}",
            "A deep dive into the {type} fiber's dynamics in the {tract}",
            "The {type} fiber's contribution to {tract} tract functionality",
            "Mapping the {type} fiber's trajectory in the {tract} on the {side} side",
            "Navigating the intricate pathways of the {type} fiber within the {tract}",
            "The interplay of {type} fibers across the {side} hemisphere",
            "Traversing the {tract} with a {type} fiber",
            "The pivotal role of the {type} fiber in connecting the {tract}",
            "Showcasing the unique texture of {type} fibers in the {tract}",
            "Zooming in on the {type} fiber's impact on the {side} hemisphere",
            "The {type} fiber in the {tract}",
            "The {type} fiber as a conduit in the {tract} on the {side} side",
            "The {type} fiber's architectural marvel within the {tract}",
            "A journey alongside the {type} fiber through the {tract}",
            "The harmonious structure of the {type} fiber in the {tract}",
            "Unraveling the secrets of the {type} fiber in the {tract} tract",
            "The {type} fiber: A key player in {tract} dynamics",
            "Envisioning the {type} fiber's pathway in the {tract}",
            "The strategic placement of the {type} fiber in the {tract}",
            "Illuminating the {type} fiber's route through the {tract}",
            "The {type} fiber: An essential bridge within the {tract}",
            "Deciphering the network of {type} fibers in the {tract}",
            "Exploring the synergy between {type} fibers and the {tract}",
            "The {type} fiber's vital link in the neural network of the {tract}",
            "The {type} fiber's role in the {tract} on the {side} side",
            "The {type} fiber's intricate design within the {tract}",
            "The {type} fiber's impact on the {tract} on the {side} side",
            "The {type} fiber's influence on the {tract}",
            "The {type} fiber's significance in the {tract}",
            "The {type} fiber's contribution to the {tract}",
            "The {type} fiber's role in the {tract}",
            "The {type} fiber's function in the {tract}",
            "The {type} fiber's purpose in the {tract}",
            "The {type} fiber's importance in the {tract}",
            "A fiber of the {tract} tract located on the {side} side",
            "A {type} fiber located on the {side} side of the {tract} tract",
            "A {type} fiber located on the {side} side of the {tract}",
            "A {type} fiber located on the {side} side",
            "{type}",
            "{tract}",
            "{side} side",
            "{type}, {side}, {tract}",
            "{type}, {tract}",
            "{type}, {side}",
            "{tract}, {type}, {side}",
            "In-depth analysis of the {type} fiber's architecture in the {tract}",
            "The {type} fiber's pivotal presence in the {tract} on the {side} side",
            "A {type} fiber's critical role in the {tract} tract's network",
            "Understanding the {type} fiber within the {tract}'s framework",
            "The {tract} tract's {type} fiber: A closer examination",
            "Insights into the {type} fiber's influence in the {tract}",
            "The {type} fiber's unique characteristics in the {tract}",
            "Exploring the {type} fiber's presence in the {tract} on the {side}",
            "The {type} fiber's essential functions within the {tract}",
            "Tracing the path of the {type} fiber through the {tract}",
            "The {type} fiber's strategic importance to the {tract}",
            "Investigating the {type} fiber's role in the {tract} tract's ecosystem",
            "The {type} fiber's contribution towards {tract} integrity",
            "Evaluating the {type} fiber's impact on the {tract} tract",
            "The {type} fiber's connectivity within the {tract}",
            "The {type} fiber and its significance to the {tract}'s structure",
            "A {type} fiber's journey across the {tract} on the {side} side",
            "The dynamic role of the {type} fiber in the {tract}",
            "Showcasing the {type} fiber within the {tract}'s complex network",
            "The {type} fiber's integration into the {tract} tract",
            "Dissecting the {type} fiber's function in the {tract}",
            "The {type} fiber's coordination with the {tract} tract",
            "A {type} fiber's influence on the {tract}'s functionality",
            "The {type} fiber's adaptation in the {tract} environment",
            "Charting the {type} fiber's relevance to the {tract}'s health",
            "The {type} fiber's orchestration within the {tract} tract on the {side}",
            "The {type} fiber's synergy with the {tract}'s architecture",
            "Unpacking the {type} fiber's functionality in the {tract}",
            "The {type} fiber within the {tract}: A study of precision",
            "The nuanced role of the {type} fiber in the {tract} tract",
            "A {type} fiber's impact on {tract} tract dynamics on the {side}",
            "The {type} fiber's contribution to the {tract}'s resilience",
            "A {type} fiber within the {tract}: Bridging critical gaps",
            "The {type} fiber's alignment with the {tract}'s objectives",
            "The {type} fiber's facilitation of communication in the {tract}",
            "Dissecting the {type} fiber's efficiency within the {tract}",
            "The {type} fiber's strategic placement in the {tract} for {side} hemisphere optimization",
            "Exploring the {type} fiber's adaptability in the {tract}",
            "The {type} fiber's role in enhancing the {tract}'s performance",
            "A {type} fiber's precision in navigating the {tract}",
            "The {type} fiber's contribution to the {tract}'s complexity",
            "A {type} fiber's resilience within the {tract} structure",
            "The {type} fiber: A cornerstone of the {tract}'s anatomy",
            "The {type} fiber's influence on {tract} tract health and functionality",
            "A {type} fiber's strategic role in the {tract}'s network on the {side}",
            "The {type} fiber's capacity for innovation in the {tract}",
            "Mapping the {type} fiber's influence across the {tract}",
            "The {type} fiber's essential service to the {tract}'s harmony",
            "The {type} fiber's role in the {tract}'s structural integrity",
            "A {type} fiber's signature within the {tract}: Unveiling its essence",
            "Profiling the {type} fiber: Key to the {tract}'s efficacy",
            "The {type} fiber as a linchpin in the {tract}'s neural circuitry",
            "Elevating {tract} tract insights: The {type} fiber's pivotal role",
            "A {type} fiber's blueprint within the {tract}'s neural landscape",
            "The {type} fiber: Sculpting the {tract}'s neuroanatomy",
            "Crafting connectivity: The {type} fiber's role in the {tract}",
            "The {type} fiber's imprint on the {tract}'s neural pathways",
            "A {type} fiber's strategic influence on {tract} tract coherence",
            "The {type} fiber's legacy within the {tract}'s neural networks",
            "A {type} fiber's mastery in shaping the {tract}",
            "Decoding the {type} fiber's essence in the {tract}'s framework",
            "The {type} fiber's orchestration of neural signals in the {tract}",
            "A {type} fiber's harmonious integration into the {tract}",
            "The {type} fiber as a navigator in the {tract}'s neural seas",
            "Crafting the {tract}'s narrative: The {type} fiber's role",
            "The {type} fiber's legacy in the {tract}'s functional architecture",
            "A {type} fiber's resonance within the {tract}'s neural harmony",
            "The {type} fiber: A beacon in the {tract}'s neural odyssey",
            "Navigating the {tract} with precision: The role of the {type} fiber",
            "The {type} fiber's symphony within the {tract}'s neural orchestra",
            "A {type} fiber's voyage through the {tract}'s neural corridors",
            "The {type} fiber: Crafting the backbone of the {tract}",
            "A {type} fiber's influence on the {tract}'s neural tapestry",
            "The {type} fiber as the {tract}'s architect: Designing neural pathways",
            "Unraveling the {type} fiber's narrative within the {tract}'s neural saga",
            "The {type} fiber's orchestral role in the neural symphony of the {tract}",
            "Navigating the nuances of the {tract} with the {type} fiber as our guide",
            "The {type} fiber: Bridging neural divides within the {tract}",
            "A {type} fiber's journey through the neural landscapes of the {tract}",
            "The {type} fiber: A neural harbinger in the {tract}'s evolution",
            "Architects of the brain: The {type} fiber's role in constructing the {tract}",
            "The {type} fiber's silent symphony within the {tract}'s neural network",
            "A {type} fiber's resilient passage through the {tract}'s neural terrain",
            "The {type} fiber: Shaping the {tract}'s neural destiny",
            "Unveiling the {type} fiber's neural tapestry in the {tract}",
            "The {type} fiber's dance across the neural stages of the {tract}",
            "A {type} fiber's imprint on the {tract}'s neural blueprint",
            "The {type} fiber: A conduit of connectivity in the {tract}",
            "Exploring the {tract}'s neural depth through the {type} fiber",
            "A {type} fiber's narrative within the neural anthology of the {tract}",
            "The {type} fiber's role in weaving the neural fabric of the {tract}",
            "A {type} fiber's echo in the neural chambers of the {tract}",
            "The {type} fiber as a neural pioneer in the {tract}'s frontier",
            "A {type} fiber's influence on the {tract}'s neural harmony",
            "The {type} fiber: A neural sentinel in the {tract}'s landscape",
            "Charting the {type} fiber's course in the {tract}'s neural voyage",
            "The {type} fiber's contribution to the {tract}'s neural mosaic",
            "A {type} fiber's legacy within the {tract}'s neural archives",
            "The {type} fiber: An architect of the {tract}'s neural pathways",
            "A {type} fiber's role in the {tract}'s neural symposium"
        ]
    
    
    # Extraer los labels de todos los grafos en el lote
    labels = [graph.y.item() for graph in batch]  # Asumiendo que `y` es el tensor de labels
    
    # Recuperar y tokenizar todos los captions necesarios en una sola llamada
    captions = [random.choice(caption_templates).format(**TRACT_LIST[LABELS[label]]) for label in labels]
    tokenized_texts_batch = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    

    return GeoBatch.from_data_list(batch), {'input_ids': tokenized_texts_batch['input_ids'], 'attention_mask': tokenized_texts_batch['attention_mask']}


def collate_function_v2(batch):
    """Funcion para el DataLoader"""

    

    tokenizer = AutoTokenizer.from_pretrained(CFG.graph_encoder_name, model_max_length=128)#"bert-base-uncased"

    TRACT_LIST = {
        "AF_L": {
            "id": 0,
            "tract": "arcuate fasciculus",
            "side": "left",
            "type": "association",
            "functional_significance": [
                "crucial for language production and comprehension, facilitating complex linguistic processing",
                "connects Broca's area and Wernicke's area, essential for the integration of expressive and receptive language functions",
                "plays a significant role in phonological processing, allowing for the manipulation of speech sounds",
                "involved in language learning and acquisition, contributing to the development of linguistic skills over time",
                "supports verbal working memory, critical for the temporary storage and manipulation of information"
            ],
            "structural_characteristics": [
                "characterized by a dense bundle of axons, indicating high connectivity between language centers",
                "shows significant left-right asymmetry, with the left side being more prominent in most individuals",
                "myelination level is high, facilitating fast signal transmission crucial for real-time language processing",
                "varies in size and density among individuals, which may correlate with language proficiency and cognitive abilities",
                "structural integrity is crucial for normal language function, with damage leading to specific aphasia types"
            ],
            "vulnerability_to_disease": [
                "vulnerable to aphasia following stroke, particularly in areas supplied by the middle cerebral artery",
                "degenerative diseases like Alzheimer's can affect its integrity, leading to progressive language deficits",
                "susceptible to traumatic brain injury impacts, which can disrupt language processing and production",
                "conditions like primary progressive aphasia specifically target language networks including the arcuate fasciculus",
                "multiple sclerosis lesions can impair its function, affecting verbal fluency and language comprehension"
            ],
        }, "AF_R": {
            "id": 1,
            "tract": "arcuate fasciculus",
            "side": "right",
            "type": "association",
            "functional_significance": [
                "supports non-verbal communication, including the processing of musical and environmental sounds",
                "contributes to emotional tone and prosody in speech, crucial for expressing and interpreting emotions",
                "involved in the integration of auditory and visual cues in communication, enhancing social interaction",
                "plays a role in spatial processing and navigation, complementing linguistic spatial descriptions",
                "facilitates creative and abstract thinking by connecting disparate cognitive and perceptual areas"
            ],
            "structural_characteristics": [
                "generally smaller and less dense than its counterpart on the left, reflecting functional specialization",
                "contains axons that are crucial for the transmission of non-linguistic auditory information",
                "the degree of myelination supports efficient processing of complex auditory signals",
                "structural variations can influence individual abilities in music perception and emotional empathy",
                "its integrity is essential for the holistic understanding of speech, including tone and emotional context"
            ],
            "vulnerability_to_disease": [
                "damage can lead to deficits in emotional expression and comprehension, impacting social communication",
                "right hemisphere strokes involving the AF can disrupt non-verbal aspects of communication",
                "degenerative conditions may lead to changes in social behavior and emotional processing",
                "traumatic injuries affecting this region can impair the perception of music and environmental sounds",
                "neurological conditions can affect spatial processing and navigation, linked to right AF functionality"
            ],
        }, "CC_Fr_1": {
            "id": 2,
            "tract": "corpus callosum, frontal lobe",
            "side": "most anterior part of the frontal lobe",
            "type": "commissural",
            "functional_significance": [
                "facilitates interhemispheric communication between the frontal lobes, crucial for executive functions",
                "supports coordination of complex cognitive tasks involving planning, decision-making, and problem-solving",
                "integral for the bilateral coordination of motor functions, affecting tasks requiring fine motor skills",
                "plays a role in the integration of sensory information across hemispheres, contributing to spatial awareness",
                "involved in the processing and regulation of emotions, linking cognitive and affective responses"
            ],
            "structural_characteristics": [
                "Composed of densely packed myelinated fibers, enabling efficient cross-hemispheric communication.",
                "Thickness and fiber density can vary, reflecting individual differences in cognitive and motor abilities.",
                "Structural integrity is crucial for synchronous activity and coordination between the hemispheres.",
                "Variability in the anterior corpus callosum size may correlate with differences in executive function.",
                "The anterior region's architecture supports a high bandwidth of information exchange."
            ],
            "vulnerability_to_disease": [
                "Susceptible to demyelinating diseases such as multiple sclerosis, affecting communication efficiency.",
                "Can be impacted by traumatic brain injury, leading to deficits in executive functions and motor coordination.",
                "Degenerative diseases like Alzheimer's can lead to atrophy, affecting cognitive processing and emotional regulation.",
                "Conditions such as corpus callosum agenesis result in developmental abnormalities in interhemispheric communication.",
                "Stroke in areas supplying blood to the corpus callosum can disrupt its function, affecting frontal lobe tasks."
            ],
        }, "CC_Fr_2": {
            "id": 3,
            "tract": "corpus callosum, frontal lobe",
            "side": "most posterior part of the frontal lobe",
            "type": "commissural",
            "functional_significance": [
                "Plays a crucial role in the integration and coordination of cognitive functions between hemispheres.",
                "Supports complex thought processes by enabling cross-hemispheric collaboration in problem-solving.",
                "Facilitates the sharing of higher order sensory processing and attentional control information.",
                "Important for bilateral motor planning and execution, impacting activities requiring coordination.",
                "Contributes to the emotional and social cognition by integrating affective information across hemispheres."
            ],
            "structural_characteristics": [
                "Features a complex network of fibers allowing for specialized communication pathways.",
                "Structural variations in this region may influence individual cognitive and motor capabilities.",
                "The posterior aspect's fiber density supports the integration of high-level cognitive functions.",
                "Adaptable and plastic, reflecting changes in cognitive demands and learning over time.",
                "Myelination patterns in this area facilitate rapid interhemispheric transfer of complex information."
            ],
            "vulnerability_to_disease": [
                "At risk for impacts from neurodegenerative conditions, affecting cognitive and motor function.",
                "Traumatic impacts can lead to disconnections, manifesting as coordination and processing deficits.",
                "Agenesis or hypogenesis can lead to developmental delays and cognitive impairments.",
                "Affected by stroke or lesions, potentially leading to disruptions in attention and executive functioning.",
                "Subject to changes in conditions like schizophrenia, impacting cognitive and emotional processing."
            ],
        }, "CC_Oc": {
            "id": 4,
            "tract": "corpus callosum, occipital lobe",
            "side": "central",
            "type": "commissural",
            "functional_significance": [
                "Essential for visual information integration, allowing for coordinated visuospatial processing.",
                "Facilitates bilateral visual field processing, contributing to depth perception and visual coherence.",
                "Supports the coordination of visual-motor responses, important for tasks requiring hand-eye coordination.",
                "Involved in the synchronization of sensory and perceptual information across the visual cortex.",
                "Plays a role in visual memory, enabling the integration of visual experiences from both hemispheres."
            ],
            "structural_characteristics": [
                "Contains densely packed fibers crucial for high-speed transmission of visual information.",
                "The occipital portion of the corpus callosum is specialized for the transfer of complex visual signals.",
                "Variations in thickness and fiber density may relate to individual differences in visual processing abilities.",
                "Myelination in this region is optimized for rapid processing and integration of visual stimuli.",
                "Structural integrity is key for the accurate and efficient bilateral coordination of visual tasks."
            ],
            "vulnerability_to_disease": [
                "Vulnerable to conditions that impair visual processing, affecting spatial awareness and perception.",
                "Damage from traumatic brain injury can disrupt visual integration, leading to difficulties in visual coherence.",
                "Degenerative diseases affecting the occipital lobe can impair bilateral visual information processing.",
                "Lesions in this area can lead to deficits in visual memory and the integration of visual fields.",
                "Affected by developmental disorders, potentially leading to atypical visual processing and perception."
            ],
        }, "CC_Pa": {
            "id": 5,
            "tract": "corpus callosum, parietal lobe",
            "side": "central",
            "type": "commissural",
            "functional_significance": [
                "Crucial for integrating sensory and motor information across hemispheres, enhancing spatial awareness and navigation.",
                "Facilitates interhemispheric transfer of tactile information, important for the perception of touch and texture.",
                "Supports coordination of complex movements, allowing for smooth execution of bilateral and unilateral motor tasks.",
                "Plays a role in visuospatial attention, enabling the brain to process and respond to stimuli in the environment effectively.",
                "Involved in the integration of language and numerical processing, contributing to complex cognitive abilities."
            ],
            "structural_characteristics": [
                "Composed of densely packed fibers, facilitating efficient communication between parietal areas of both hemispheres.",
                "The parietal portion of the corpus callosum supports the high bandwidth necessary for the transfer of complex sensory information.",
                "Variability in thickness and fiber density can reflect individual differences in sensory processing and motor coordination abilities.",
                "Myelination patterns within this segment are optimized for the rapid relay of somatosensory and motor signals.",
                "The integrity of this structure is essential for coordinating spatial processing and sensorimotor integration."
            ],
            "vulnerability_to_disease": [
                "Susceptible to the effects of neurodegenerative diseases, impacting spatial reasoning and sensorimotor functions.",
                "Traumatic injuries affecting this area can lead to disorders in tactile perception and motor coordination.",
                "Lesions can disrupt the interhemispheric transfer of information, affecting spatial awareness and navigation.",
                "Affected by conditions like stroke, which can impair the connectivity and function of the parietal lobes.",
                "Developmental disorders may lead to atypical wiring, influencing sensory processing and spatial cognition."
            ],
        }, "CC_Pr_Po": {
            "id": 6,
            "tract": "corpus callosum, pre/post central gyri",
            "side": "central",
            "type": "commissural",
            "functional_significance": [
                "Integral for the coordination of sensory and motor functions, facilitating bilateral motor control.",
                "Enables the transfer of motor planning and execution signals, critical for coordinated movements.",
                "Supports somatosensory integration, allowing for the perception of touch across the body's bilateral regions.",
                "Plays a key role in the bilateral synchronization of movements, enhancing motor skills and dexterity.",
                "Involved in the integration of sensory feedback into motor actions, important for adaptive motor control."
            ],
            "structural_characteristics": [
                "Features a dense network of fibers connecting motor and sensory areas, ensuring efficient communication.",
                "The structural design supports the rapid transmission of motor and sensory signals across hemispheres.",
                "Myelination is optimized for high-speed signal relay, crucial for timely motor responses and sensory processing.",
                "Variations in this region's architecture can influence motor coordination and sensory perception abilities.",
                "The integrity of this segment is crucial for the seamless execution of complex motor tasks and sensory integration."
            ],
            "vulnerability_to_disease": [
                "Vulnerable to motor control disorders resulting from disruptions in interhemispheric communication.",
                "Injuries or lesions here can lead to deficits in tactile sensation and proprioception, impacting daily activities.",
                "Degenerative diseases affecting this region can impair motor function and somatosensory processing.",
                "Affected by stroke, leading to challenges in motor coordination and the integration of sensory information.",
                "Developmental conditions may alter the connectivity patterns, affecting motor skills and sensory experiences."
            ],
        }, "CG_L": {
            "id": 7,
            "tract": "cingulum",
            "side": "left",
            "type": "association",
            "functional_significance": [
                "Supports emotional regulation and processing, linking limbic structures involved in mood and affect.",
                "Crucial for cognitive control and attention, facilitating executive functions and working memory.",
                "Involved in the consolidation and retrieval of memories, connecting hippocampal and cortical structures.",
                "Plays a role in spatial orientation and navigation, integrating spatial memory and environmental cues.",
                "Supports language processing and semantic memory, contributing to comprehension and verbal memory."
            ],
            "structural_characteristics": [
                "Consists of a long, curved bundle of fibers running above the corpus callosum, connecting frontal, parietal, and temporal lobes.",
                "Densely packed axons facilitate efficient communication between cognitive and emotional processing centers.",
                "Variability in thickness and tract integrity may correlate with individual differences in emotional and cognitive capacities.",
                "Myelinated fibers ensure rapid transmission of signals crucial for the integration of cognitive and emotional information.",
                "Structural integrity is vital for maintaining cognitive functions and emotional stability."
            ],
            "vulnerability_to_disease": [
                "Affected by Alzheimer's disease, leading to memory impairments and emotional disturbances.",
                "Vulnerable to traumatic brain injury, which can disrupt cognitive and emotional processing.",
                "Degeneration can contribute to anxiety and mood disorders, reflecting its role in emotional regulation.",
                "Lesions can impair memory formation and retrieval, affecting learning and spatial navigation.",
                "Subject to changes in connectivity in psychiatric conditions, influencing cognitive and emotional functions."
            ],
        }, "CG_R": {
            "id": 8,
            "tract": "cingulum",
            "side": "right",
            "type": "association",
            "functional_significance": [
                "Plays a crucial role in the integration of emotional and spatial information, supporting right-hemisphere dominance for spatial and emotional processing.",
                "Involved in attentional processes, particularly in the spatial and visual domains, facilitating awareness of environmental stimuli.",
                "Supports nonverbal memory processes, including the retrieval of visual and spatial memories.",
                "Contributes to the regulation of emotions and stress responses, linking to limbic structures.",
                "Facilitates the integration of sensory information into emotionally relevant contexts, enhancing emotional and social communication."
            ],
            "structural_characteristics": [
                "Mirrors the left cingulum in structure but supports functions more prevalent in the right hemisphere, including spatial and emotional processing.",
                "The fiber density and myelination pattern are optimized for the integration and processing of nonverbal, emotional, and spatial information.",
                "Variability in the right cingulum's structure may be associated with individual differences in spatial abilities and emotional regulation.",
                "Essential for connecting right hemisphere brain regions involved in emotional and spatial cognition.",
                "Its structural integrity is crucial for the coordination and integration of right hemisphere functions."
            ],
            "vulnerability_to_disease": [
                "Alterations in the right cingulum have been associated with emotional dysregulation and spatial processing deficits.",
                "Traumatic brain injuries affecting the right side can impact spatial awareness and emotional comprehension.",
                "Degenerative conditions can lead to impairments in nonverbal memory and spatial orientation.",
                "Lesions in this area may disrupt the emotional response to sensory stimuli, affecting emotional well-being.",
                "Psychiatric conditions may alter its connectivity, influencing emotional processing and stress responses."
            ],
        }, "FAT_L": {
            "id": 9,
            "tract": "frontal aslant tract",
            "side": "left",
            "type": "association",
            "functional_significance": [
            "Integral for speech production and language articulation, connecting Broca's area to the supplementary motor area.",
            "Supports complex motor planning and execution, crucial for verbal fluency.",
            "Involved in cognitive control and decision-making processes, facilitating executive functions.",
            "Plays a role in working memory, particularly in the manipulation of verbal information.",
            "Contributes to the regulation of attention, enhancing task-oriented focus and cognitive flexibility."
            ],
            "structural_characteristics": [
                "Consists of a bundle of fibers connecting frontal cortex regions, characterized by its slanted orientation.",
                "Vital for mediating the interactions between language production areas and motor planning regions.",
                "The left FAT is typically more prominent in right-handed individuals, reflecting lateralization of language functions.",
                "Myelinated fibers ensure efficient signal transmission, critical for rapid language articulation and motor coordination.",
                "Structural integrity is crucial for speech production, with variations linked to individual differences in language abilities."
            ],
            "vulnerability_to_disease": [
                "Susceptible to the effects of stroke, leading to aphasia or speech production difficulties.",
                "Degenerative diseases can impact its function, affecting verbal fluency and executive control.",
                "Traumatic brain injuries may disrupt its integrity, leading to deficits in motor planning and language processing.",
                "Conditions such as primary progressive aphasia may target the FAT, resulting in progressive loss of speech and language skills.",
                "Neurodevelopmental disorders could alter its development, affecting language acquisition and cognitive functions."
            ],
        }, "FAT_R": {
            "id": 10,
            "tract": "frontal aslant tract",
            "side": "right",
            "type": "association",
            "functional_significance": [
                "Supports non-verbal aspects of communication, including prosody and emotional expression in speech.",
                "Involved in the coordination of complex motor actions, contributing to non-verbal communication skills.",
                "Plays a role in spatial awareness and attention, particularly in tasks requiring motor response.",
                "Contributes to cognitive flexibility and inhibitory control, facilitating adaptive behavior.",
                "Supports the integration of sensory and motor information, enhancing overall motor coordination."
            ],
            "structural_characteristics": [
                "Mirrors the left FAT in structure but is involved in functions more prevalent in the right hemisphere.",
                "Characterized by connectivity between the right frontal cortex and motor areas, supporting non-verbal communication.",
                "The right FAT's structural variations can influence spatial and motor abilities, as well as emotional processing.",
                "Optimized for the coordination of right hemisphere-dominated tasks, including spatial processing and attention.",
                "Structural integrity supports non-verbal aspects of communication and motor control."
            ],
            "vulnerability_to_disease": [
                "Lesions or damage can affect emotional expression and recognition, impacting social interactions.",
                "Degenerative conditions impacting the right FAT may lead to deficits in non-verbal communication and spatial awareness.",
                "Traumatic injuries affecting this area can disrupt motor coordination and spatial orientation.",
                "Stroke affecting the right frontal region may lead to changes in emotional processing and prosody in speech.",
                "Developmental conditions may influence its formation, affecting non-verbal communication and executive functions."
            ],
        }, "FPT_L": {
            "id": 11,
            "tract": "fronto-pontine tract",
            "side": "left",
            "type": "association",
            "functional_significance": [
                "Facilitates the transmission of motor signals from the frontal cortex to the pons, coordinating voluntary movements.",
                "Supports the planning and initiation of complex motor tasks, integrating cognitive and motor processes.",
                "Involved in the regulation of fine motor skills, affecting precision and coordination in tasks.",
                "Plays a role in the modulation of attention, particularly in tasks requiring coordinated motor responses.",
                "Contributes to learning motor skills through the integration of sensory feedback and motor planning."
            ],
            "structural_characteristics": [
                "Composed of descending fibers that connect the frontal cortex to the pons, crucial for motor control.",
                "The left side is particularly involved in the coordination of right-sided voluntary movements.",
                "Myelination ensures rapid conduction of motor signals, facilitating timely and precise motor responses.",
                "Structural integrity is essential for the execution of coordinated movements and motor planning.",
                "Variability in tract structure may influence individual motor abilities and skill learning."
            ],
            "vulnerability_to_disease": [
                "Vulnerable to stroke, leading to motor deficits and impairments in voluntary movement control.",
                "Degenerative motor neuron diseases can disrupt signal transmission, affecting motor function.",
                "Traumatic brain injury may damage the tract, leading to challenges in motor coordination and execution.",
                "Lesions affecting the left FPT can result in unilateral motor control deficits, impacting daily activities.",
                "Developmental disorders affecting the tract's formation may lead to motor planning and coordination issues."
            ],
        }, "FPT_R": {
            "id": 12,
            "tract": "fronto-pontine tract",
            "side": "right",
            "type": "association",
            "functional_significance": [
                "Key for transmitting motor commands from the right frontal cortex to the pons, essential for left-sided voluntary movements.",
                "Supports the coordination and execution of movements, integrating motor signals with sensory feedback.",
                "Involved in the regulation of bilateral motor tasks, contributing to the synchronization of limb movements.",
                "Plays a role in spatially oriented motor tasks, requiring precise motor control and planning.",
                "Facilitates cognitive aspects of motor control, linking decision-making processes to motor execution."
            ],
            "structural_characteristics": [
                "Consists of descending pathways crucial for the relay of motor signals to the pontine nuclei.",
                "The right FPT is essential for the control of movements on the contralateral (left) side of the body.",
                "Optimized myelination patterns facilitate efficient motor signal transmission for rapid response.",
                "Structural integrity is vital for the precise execution and coordination of complex motor tasks.",
                "Individual differences in the structure can reflect variability in motor skills and coordination."
            ],
            "vulnerability_to_disease": [
                "Susceptible to impacts from neurological conditions that impair motor function and coordination.",
                "Stroke affecting this region can result in contralateral motor deficits, impacting fine motor skills.",
                "Traumatic injuries to the right frontal cortex can disrupt the FPT, leading to motor planning difficulties.",
                "Degenerative diseases may affect the integrity of the tract, resulting in progressive motor skill loss.",
                "Developmental anomalies in the tract's formation can lead to challenges in motor learning and execution."
            ],
        }, "FX_L": {
            "id": 13,
            "tract": "fornix",
            "side": "left",
            "type": "commissural",
            "functional_significance": [
                "Crucial for hippocampal connectivity, supporting memory formation, consolidation, and retrieval.",
                "Facilitates the transmission of signals related to spatial memory and navigation.",
                "Involved in the regulation of emotional responses, linking hippocampal and limbic system functions.",
                "Supports learning processes, particularly in the context of associative and contextual memory.",
                "Plays a role in the modulation of attention, particularly in tasks requiring memory integration."
            ],
            "structural_characteristics": [
                "Composed of a bundle of fibers that arch over the thalamus, connecting hippocampal regions to other limbic structures.",
                "The left fornix is integral to the limbic system, supporting emotional and memory-related processes.",
                "Myelinated fibers ensure the efficient transmission of neural signals critical for memory and learning.",
                "Structural integrity is essential for the maintenance of long-term memory and emotional regulation.",
                "Variations in the fornix's structure can influence individual differences in memory capacity and emotional stability."
            ],
            "vulnerability_to_disease": [
                "Vulnerable to degenerative diseases like Alzheimer's, leading to memory impairments and spatial disorientation.",
                "Affected by traumatic brain injury, which can disrupt memory formation and emotional responses.",
                "Lesions in the fornix can result in anterograde amnesia, affecting the ability to form new memories.",
                "Neurodevelopmental conditions may alter its development, impacting memory and emotional processing.",
                "Susceptible to the effects of stress and depression, potentially leading to changes in structure and function."
            ],
        }, "FX_R": {
            "id": 14,
            "tract": "fornix",
            "side": "right",
            "type": "commissural",
            "functional_significance": [
                "Supports the right hippocampal functions, contributing to spatial and episodic memory processes.",
                "Facilitates the integration of memory and emotional information from the right hemisphere.",
                "Plays a role in the navigation and orientation in space, supporting right hemisphere-dominated spatial tasks.",
                "Involved in the regulation of emotional and stress responses, contributing to overall emotional well-being.",
                "Supports associative learning and memory consolidation, particularly for non-verbal and spatial information."
            ],
            "structural_characteristics": [
                "Mirrors the left fornix in its role but supports functions associated with the right hippocampal region.",
                "Comprised of myelinated fibers that facilitate communication between the hippocampus and limbic structures.",
                "Structural integrity on the right side is crucial for spatial memory and emotional processing.",
                "Variations in structure can reflect individual differences in spatial abilities and emotional regulation.",
                "The fornix's architecture is optimized for the rapid relay of information critical for memory and learning."
            ],
            "vulnerability_to_disease": [
                "Neurodegenerative diseases affecting the right fornix can impair spatial memory and emotional stability.",
                "Traumatic injuries to the right side can disrupt spatial orientation and episodic memory retrieval.",
                "Lesions may lead to deficits in non-verbal memory, affecting the recall of spatial and visual information.",
                "Developmental disorders impacting the fornix can influence spatial learning and emotional processing skills.",
                "Stress and psychological conditions can affect its structure, potentially altering memory and emotional responses."
            ],
        }, "IFOF_L": {
            "id": 15,
            "tract": "inferior fronto-occipital fasciculus",
            "side": "left",
            "type": "association",
            "functional_significance": [
                "Facilitates complex visual processing and integration with language functions.",
                "Supports reading and comprehension by connecting visual and language areas.",
                "Involved in attentional control and working memory, particularly for visual and spatial tasks.",
                "Plays a role in the processing of semantic information, essential for understanding and language production.",
                "Contributes to the coordination of visually guided movements, linking perception to action."
            ],
            "structural_characteristics": [
                "A long association fiber tract connecting the frontal and occipital lobes, with extensions to the temporal lobe.",
                "Supports efficient communication between areas involved in language, visual processing, and executive functions.",
                "The left IFOF is particularly important for language and semantic processing, reflecting lateralization of these functions.",
                "Highly myelinated, ensuring rapid signal transmission across distant brain regions involved in complex cognitive tasks.",
                "Variability in tract integrity and connectivity may correlate with individual differences in language and visual processing abilities."
            ],
            "vulnerability_to_disease": [
                "Lesions can lead to deficits in visual processing and language comprehension, affecting reading and semantic understanding.",
                "Degenerative diseases may impact the tract's integrity, leading to progressive cognitive decline in related functions.",
                "Traumatic brain injury can disrupt the IFOF, resulting in difficulties with attention, memory, and executive functions.",
                "Affected by stroke, potentially leading to alexia or agraphia, depending on the specific location of damage.",
                "Developmental variations in the IFOF may contribute to conditions like dyslexia, affecting reading and language acquisition."
            ],
        }, "IFOF_R": {
            "id": 16,
            "tract": "inferior fronto-occipital fasciculus",
            "side": "right",
            "type": "association",
           "functional_significance": [
                "Supports non-verbal and visuospatial processing, integrating visual information with executive functions.",
                "Involved in the recognition and processing of visual patterns and spatial orientation.",
                "Plays a role in attentional control and the integration of sensory information for spatial awareness.",
                "Contributes to the processing of visual emotions and social cues, important for social communication.",
                "Facilitates the coordination of visually guided movements in spatial and non-verbal tasks."
            ],
            "structural_characteristics": [
                "Connects frontal and occipital lobes, with extensions to the temporal and parietal areas, supporting visuospatial integration.",
                "Essential for the right hemisphere's dominance in spatial processing and visual pattern recognition.",
                "The right IFOF's myelination supports efficient processing of complex visuospatial information.",
                "Structural variations may influence abilities in spatial reasoning, navigation, and perception of visual art.",
                "Integrity of the tract is crucial for the holistic integration of visual and spatial information."
            ],
            "vulnerability_to_disease": [
                "Damage can result in difficulties with spatial reasoning, visual pattern recognition, and navigation.",
                "Impacted by neurological diseases affecting right hemisphere functions, leading to visuospatial deficits.",
                "Traumatic injuries to the right IFOF can disrupt visual processing and attention to the visual field.",
                "Stroke involving the right frontal-occipital regions can impair non-verbal reasoning and spatial awareness.",
                "Developmental issues affecting the tract may contribute to challenges in non-verbal learning and spatial tasks."
            ],
        }, "ILF_L": {
            "id": 17,
            "tract": "inferior longitudinal fasciculus",
            "side": "left",
            "type": "association",
            "functional_significance": [
                "Important for the integration of visual and linguistic information, supporting reading and visual comprehension.",
                "Facilitates the processing of visual language, such as reading facial expressions and body language.",
                "Involved in memory retrieval, particularly visual memories associated with language.",
                "Supports language development by linking visual input to language processing areas.",
                "Plays a role in semantic processing, enabling the understanding of symbols and written language."
            ],
            "structural_characteristics": [
                "Runs longitudinally along the temporal lobe, connecting occipital and temporal lobes, crucial for visual-language integration.",
                "The left ILF supports language-related visual processing, reflecting the brain's functional lateralization.",
                "Myelinated fibers ensure efficient communication between visual processing and language areas.",
                "Structural integrity is vital for reading and interpreting complex visual-language tasks.",
                "Variability in the ILF may correlate with individual differences in language-based visual processing skills."
            ],
            "vulnerability_to_disease": [
                "Lesions can lead to alexia or difficulties in visual word recognition, impacting reading abilities.",
                "Degenerative conditions affecting the ILF can result in progressive loss of language-related visual functions.",
                "Traumatic brain injury may disrupt its function, leading to challenges in visual memory and language comprehension.",
                "Impacted by developmental disorders, potentially contributing to dyslexia or other language acquisition issues.",
                "Affected by stroke, potentially leading to deficits in visual processing of language and semantic information."
            ],
        }, "ILF_R": {
            "id": 18,
            "tract": "inferior longitudinal fasciculus",
            "side": "right",
            "type": "association",
            "functional_significance": [
                "Supports the integration of visual and emotional content, important for interpreting emotional expressions.",
                "Facilitates visuospatial processing and navigation, contributing to spatial orientation and movement.",
                "Involved in the processing of non-verbal cues, such as facial recognition and body language interpretation.",
                "Plays a role in visual memory, particularly for emotionally charged or spatially relevant information.",
                "Contributes to the perception of art and aesthetics, linking visual input to emotional responses."
            ],
            "structural_characteristics": [
                "Connects the occipital and temporal lobes, supporting the integration of visual and spatial information.",
                "The right ILF plays a key role in the right hemisphere's dominance for spatial and emotional processing.",
                "Myelinated fibers facilitate rapid transmission of visuospatial and emotional information.",
                "Structural integrity is crucial for the perception and interpretation of complex visual stimuli.",
                "Variations in the ILF may influence abilities in spatial reasoning and emotional intelligence."
            ],
            "vulnerability_to_disease": [
                "Damage can result in deficits in facial recognition and interpretation of non-verbal cues.",
                "Neurological conditions affecting the right ILF can impair spatial navigation and visual memory.",
                "Traumatic injuries may disrupt its function, affecting emotional processing and visuospatial skills.",
                "Affected by developmental conditions, potentially leading to challenges in social communication and orientation.",
                "Stroke involving the right temporal-occipital regions can impact emotional and visual processing."
            ],
        }, "MCP": {
            "id": 19,
            "tract": "middle cerebellar peduncle",
            "side": "central",
            "type": "commissural",
            "functional_significance": [
                "Facilitates communication between the cerebellum and the rest of the brain, supporting motor coordination and balance.",
                "Plays a crucial role in the timing and precision of movements, essential for smooth execution of motor tasks.",
                "Involved in motor learning, allowing for the adaptation and fine-tuning of movements based on sensory feedback.",
                "Supports cognitive functions related to the cerebellum, such as attention and language processing.",
                "Contributes to the regulation of voluntary movements, including gait and posture control."
            ],
            "structural_characteristics": [
                "Composed of fibers connecting the pons to the cerebellum, forming the largest cerebellar peduncle.",
                "Facilitates the high-volume exchange of motor and sensory information necessary for coordination and learning.",
                "The integrity and myelination of fibers are critical for efficient cerebellar communication and function.",
                "Variability in the MCP's structure can reflect individual differences in motor skills and learning capacity.",
                "Structural integrity is essential for the accurate execution of coordinated movements and motor skills."
            ],
            "vulnerability_to_disease": [
                "Lesions or damage can lead to ataxia, affecting coordination and precision of movements.",
                "Degenerative diseases impacting the MCP can result in progressive motor coordination deficits.",
                "Traumatic brain injury can disrupt communication, leading to difficulties with balance and motor learning.",
                "Affected by conditions such as multiple sclerosis, potentially leading to impaired motor function and coordination.",
                "Developmental disorders affecting the MCP can contribute to delays in motor skill acquisition and coordination issues."
            ],
        }, "MdLF_L": {
            "id": 20,
            "tract": "middle longitudinal fasciculus",
            "side": "left",
            "type": "association",
            "functional_significance": [
                "Supports language processing and semantic memory, facilitating the integration of auditory and visual information.",
                "Involved in the processing of speech and language, contributing to comprehension and production.",
                "Plays a role in attentional mechanisms, particularly in the auditory and visual domains.",
                "Contributes to the integration of sensory information for language, supporting narrative comprehension.",
                "Facilitates the temporal coherence of auditory and visual language inputs, essential for effective communication."
            ],
            "structural_characteristics": [
                "Connects temporal, parietal, and occipital lobes, supporting multimodal language processing.",
                "The left MdLF is particularly important for language-related functions, reflecting hemispheric specialization.",
                "Myelinated fibers ensure efficient communication between areas involved in language and sensory processing.",
                "Structural integrity is vital for language comprehension and the integration of auditory and visual information.",
                "Variability in the MdLF may correlate with individual differences in language processing and auditory-visual integration."
            ],
            "vulnerability_to_disease": [
                "Lesions can lead to language comprehension deficits, affecting the ability to process and integrate language inputs.",
                "Degenerative conditions may impact its function, leading to progressive loss of language and communication skills.",
                "Traumatic brain injury may disrupt its integrity, leading to difficulties in language processing and sensory integration.",
                "Affected by stroke, potentially leading to deficits in auditory-visual language processing and semantic memory.",
                "Developmental variations in the MdLF may contribute to conditions like developmental language disorders."
            ],
        }, "MdLF_R": {
            "id": 21,
            "tract": "middle longitudinal fasciculus",
            "side": "right",
            "type": "association",
            "functional_significance": [
                "Supports the integration of non-verbal auditory and visual information, important for spatial and environmental awareness.",
                "Involved in the processing of non-linguistic sounds and visual cues, contributing to social communication.",
                "Plays a role in attention to and integration of multimodal sensory information, particularly in the spatial domain.",
                "Contributes to the perception and interpretation of emotional content in speech and facial expressions.",
                "Facilitates the coordination of sensory inputs for spatial navigation and orientation."
            ],
            "structural_characteristics": [
                "Connects the right temporal, parietal, and occipital lobes, supporting spatial and non-verbal processing.",
                "Essential for the right hemisphere's dominance in spatial awareness and emotional processing.",
                "Myelinated fibers facilitate rapid transmission of sensory information across modalities.",
                "Structural integrity is crucial for the perception and integration of complex sensory stimuli.",
                "Variations in the MdLF may influence abilities in spatial reasoning, emotional intelligence, and sensory integration."
            ],
            "vulnerability_to_disease": [
                "Damage can result in difficulties with spatial awareness and the interpretation of non-verbal cues.",
                "Neurological diseases affecting the right MdLF can impair the processing of emotional content and spatial orientation.",
                "Traumatic injuries may disrupt its function, affecting the integration of visual and auditory information.",
                "Stroke involving the right parietal-occipital regions can impact spatial navigation and non-verbal communication.",
                "Developmental issues affecting the tract may contribute to challenges in spatial reasoning and emotional processing."
            ],
        }, "OR_ML_L": {
            "id": 22,
            "tract": "optic radiation, Meyer loop",
            "side": "left",
            "type": "projection",
            "functional_significance": [
                "Carries visual information from the lateral geniculate nucleus to the primary visual cortex.",
                "Crucial for processing the upper quadrant of the contralateral visual field.",
                "Supports visual perception, including color, motion, and shape recognition.",
                "Integral for the spatial organization and interpretation of visual stimuli.",
                "Facilitates visual awareness and contributes to visual-guided reflexes and responses."
            ],
            "structural_characteristics": [
                "Consists of axonal fibers looping anteriorly into the temporal lobe before projecting posteriorly.",
                "The left Meyer loop processes visual information from the right visual field's upper quadrant.",
                "Highly myelinated, ensuring rapid transmission of visual signals.",
                "Structural integrity is vital for accurate visual processing and perception.",
                "Variations in the loop's trajectory can affect visual field processing."
            ],
            "vulnerability_to_disease": [
                "Lesions can lead to superior quadrantanopia, affecting the upper visual field.",
                "Vulnerable to damage from temporal lobe resections, such as in epilepsy surgery.",
                "Stroke or trauma affecting the loop can impair visual perception and processing.",
                "Degenerative diseases may affect signal transmission, leading to visual impairments.",
                "Developmental abnormalities can result in altered visual field processing."
            ],
        }, "OR_ML_R": {
            "id": 23,
            "tract": "optic radiation, Meyer loop",
            "side": "right",
            "type": "projection",
            "functional_significance": [
                "Transmits visual information from the thalamus to the visual cortex.",
                "Essential for processing the left visual field's upper quadrant.",
                "Supports critical aspects of visual cognition, including spatial and motion awareness.",
                "Involved in the integration of visual inputs for comprehensive visual perception.",
                "Contributes to the coordination of eye movements and visual attention."
            ],
            "structural_characteristics": [
                "Features axons curving around the temporal horn of the lateral ventricle.",
                "The right Meyer loop is responsible for visual signals from the left visual field's upper quadrant.",
                "Optimized for fast signal propagation through high myelination.",
                "The loop's path and integrity are crucial for the visual field's spatial accuracy.",
                "Individual differences in path can influence visual processing efficiency."
            ],
            "vulnerability_to_disease": [
                "Damage can cause superior quadrantanopia in the left visual field.",
                "At risk from surgical interventions in the right temporal lobe.",
                "Traumatic brain injury may disrupt its function, leading to partial visual loss.",
                "Degenerative conditions impacting the right loop can impair visual cognition.",
                "Developmental variations can lead to unique patterns of visual field processing."
            ],
        }, "POPT_L": {
            "id": 24,
            "tract": "pontine crossing tract",
            "side": "left",
            "type": "commissural",
            "functional_significance": [
                "Facilitates cross-communication between hemispheres in the pons, particularly for motor signals.",
                "Supports coordination of bilateral movements and postural adjustments.",
                "Involved in the transmission of auditory and vestibular information across hemispheres.",
                "Contributes to the regulation of sleep and arousal states through pontine networks.",
                "Plays a role in the integration of sensory and motor pathways for reflex actions."
            ],
            "structural_characteristics": [
                "Comprises fibers crossing midline at the pontine level, linking bilateral pontine structures.",
                "Left-sided tract contributes to the coordination of right-sided motor and sensory functions.",
                "Structural configuration allows for efficient bilateral communication and coordination.",
                "The integrity of these fibers is essential for balanced and coordinated motor activity.",
                "Variability in the tract's structure can affect bilateral motor and sensory integration."
            ],
            "vulnerability_to_disease": [
                "Lesions can disrupt bilateral motor coordination, leading to asymmetrical motor function.",
                "Vulnerable to pontine strokes, which can affect motor control and sensory processing.",
                "Degenerative diseases impacting the pons can impair cross-hemispheric communication.",
                "Traumatic injury to the pons can result in deficits in coordination and balance.",
                "Developmental disorders may alter the formation of the tract, affecting motor and sensory integration."
            ],
        }, "POPT_R": {
            "id": 25,
            "tract": "pontine crossing tract",
            "side": "right",
            "type": "commissural",
            "functional_significance": [
                "Enables communication between right and left pontine regions, essential for motor balance and coordination.",
                "Supports the integration of bilateral sensory input, enhancing spatial and tactile perception.",
                "Involved in coordinating bilateral reflexes and autonomic functions.",
                "Facilitates auditory information crossing, contributing to binaural hearing.",
                "Essential for maintaining equilibrium and coordinating eye movements."
            ],
            "structural_characteristics": [
                "Contains fibers that cross the midline in the pons, connecting bilateral pontine nuclei.",
                "Right-sided fibers are key for the coordination of left-sided motor and sensory activities.",
                "Efficient bilateral coordination is ensured by the tract's structural design and connectivity.",
                "The structural integrity is crucial for symmetrical motor function and sensory processing.",
                "Differences in tract structure can influence sensory-motor integration and reflexes."
            ],
            "vulnerability_to_disease": [
                "Lesions or damage can lead to difficulties in bilateral motor coordination and sensory processing.",
                "At risk from conditions affecting the brainstem, such as stroke, impacting motor control.",
                "Degenerative brainstem diseases can affect the tract, leading to coordination and balance issues.",
                "Trauma to the pons can impair the functions mediated by the pontine crossing tracts.",
                "Developmental anomalies in the tract can result in impaired motor and sensory integration."
            ],
        }, "PYT_L": {
            "id": 26,
            "tract": "pyramidal tract",
            "side": "left",
            "type": "projection",
            "functional_significance": [
                "Principal pathway for voluntary motor commands from the cortex to the spinal cord and brainstem.",
                "Facilitates fine motor control, particularly in the contralateral (right) limbs.",
                "Critical for the execution of precise, skilled movements.",
                "Involved in motor speech production, influencing articulation and phonation.",
                "Plays a role in modulating reflexes and voluntary eye movements."
            ],
            "structural_characteristics": [
                "Consists of large, myelinated axons for rapid signal transmission.",
                "Originates in the motor cortex, descending through the internal capsule to the brainstem and spinal cord.",
                "The left tract primarily controls muscles on the body's right side due to the decussation in the medulla.",
                "Highly organized topographically, with fibers arranged according to the body part they control.",
                "Structural integrity is essential for precise and coordinated motor function."
            ],
            "vulnerability_to_disease": [
                "Lesions can result in hemiparesis or hemiplegia of the contralateral side, affecting motor control.",
                "Degenerative diseases like ALS can impair the tract's function, leading to progressive motor decline.",
                "Stroke affecting the cortical or subcortical regions can disrupt motor commands, impacting movement.",
                "Traumatic brain injury may damage the tract, leading to motor deficits.",
                "Developmental disorders can affect the formation or function of the tract, impacting motor skills."
            ],
        }, "PYT_R": {
            "id": 27,
            "tract": "pyramidal tract",
            "side": "right",
            "type": "projection",
            "functional_significance": [
                "Conveys voluntary motor signals from the right cerebral cortex to lower motor neurons.",
                "Essential for controlling fine movements in the contralateral (left) side of the body.",
                "Plays a significant role in motor aspects of speech production on the left side.",
                "Involved in the coordination and planning of movements.",
                "Affects manual dexterity and precision in tasks requiring fine motor skills."
            ],
            "structural_characteristics": [
                "Comprised of axonal fibers descending from the right motor cortex to the spinal cord.",
                "Crosses over to the opposite side at the medullary decussation, influencing left-side motor control.",
                "The tract's myelination ensures efficient and fast transmission of motor commands.",
                "Topographic organization mirrors that of the left side, maintaining precise control over specific muscles.",
                "Structural health is crucial for the smooth execution of voluntary movements."
            ],
            "vulnerability_to_disease": [
                "Damage can lead to contralateral motor impairments, affecting the left side of the body.",
                "Neurodegenerative conditions can deteriorate tract function, diminishing motor abilities.",
                "Ischemic stroke in relevant brain areas can cause acute motor control loss.",
                "Injuries to the right cerebral hemisphere may result in deficits in precise motor functions.",
                "Early developmental issues can disrupt normal motor development and control."
            ],
        }, "SLF_L": {
            "id": 28,
            "tract": "superior longitudinal fasciculus",
            "side": "left",
            "type": "association",
            "functional_significance": [
                "Facilitates communication between the frontal, parietal, and temporal lobes, supporting language processing.",
                "Integral for working memory, especially verbal working memory.",
                "Supports the integration of sensory information for language comprehension.",
                "Plays a role in attentional control and executive functions.",
                "Involved in the coordination of speech production and articulation."
            ],
            "structural_characteristics": [
                "A long bundle of fibers connecting various cortical regions across the left hemisphere.",
                "Supports the integration of language functions across spatially separated brain regions.",
                "Myelinated fibers ensure rapid communication essential for language and cognitive tasks.",
                "Structural integrity is crucial for efficient language processing and executive functioning.",
                "Variations in the SLF can reflect individual differences in language ability and cognitive processing."
            ],
            "vulnerability_to_disease": [
                "Lesions can impair language processing and production, leading to aphasia.",
                "Degenerative diseases may affect its integrity, resulting in cognitive and linguistic deficits.",
                "Traumatic brain injury can disrupt its function, impacting language and executive functions.",
                "Stroke in the left hemisphere can lead to deficits in working memory and attention.",
                "Developmental disorders may alter its formation, affecting language development and cognitive skills."
            ],
        }, "SLF_R": {
            "id": 29,
            "tract": "superior longitudinal fasciculus",
            "side": "right",
            "type": "association",
            "functional_significance": [
                "Supports spatial awareness and attention, integrating sensory and motor information.",
                "Involved in non-verbal communication, including the interpretation of gestures and facial expressions.",
                "Plays a role in visuospatial processing and navigation.",
                "Contributes to the integration of auditory and visual stimuli for spatial tasks.",
                "Facilitates right hemisphere functions in attention and executive control."
            ],
            "structural_characteristics": [
                "Connects frontal, parietal, and temporal regions in the right hemisphere, supporting spatial and non-verbal integration.",
                "Ensures efficient communication between regions involved in spatial and attentional processing.",
                "The right SLF is critical for the coordination of sensory information with motor responses.",
                "Myelination patterns support the rapid relay of information necessary for spatial cognition.",
                "The structural health of the SLF is key for effective visuospatial and attentional functions."
            ],
            "vulnerability_to_disease": [
                "Damage can lead to spatial neglect, particularly affecting attention to the contralateral (left) side.",
                "Affected by conditions that impair spatial reasoning and non-verbal communication.",
                "Traumatic injuries may impact spatial awareness and executive functions.",
                "Degenerative diseases can lead to progressive decline in spatial and attentional capabilities.",
                "Developmental variations can influence the capacity for visuospatial processing and non-verbal reasoning."
            ],
        }, "UF_L": {
            "id": 30,
            "tract": "uncinate fasciculus",
            "side": "left",
            "type": "association",
            "functional_significance": [
                "Facilitates emotional regulation by connecting the frontal lobe to the amygdala and anterior temporal lobe.",
                "Supports language comprehension and production, linking areas involved in semantic processing.",
                "Involved in memory formation and retrieval, particularly emotional memories.",
                "Plays a role in decision-making processes, integrating emotional and cognitive information.",
                "Contributes to social cognition, aiding in the interpretation of emotional cues and empathy."
            ],
            "structural_characteristics": [
                "A hook-shaped bundle of fibers connecting the orbitofrontal cortex to the anterior temporal lobe.",
                "Essential for the bidirectional transfer of information between key areas for emotion and cognition.",
                "Myelinated fibers ensure rapid communication, critical for emotional and social responses.",
                "The left UF is particularly important for language-related emotional processing.",
                "Structural integrity is crucial for maintaining emotional regulation and social interaction skills."
            ],
            "vulnerability_to_disease": [
                "Lesions can lead to difficulties in emotional regulation and decision-making.",
                "Degenerative diseases impacting the UF may result in emotional blunting or disinhibition.",
                "Affected by temporal lobe epilepsy, potentially leading to changes in emotional behavior and memory.",
                "Traumatic brain injury may disrupt its function, affecting emotional processing and social cognition.",
                "Altered connectivity has been associated with psychiatric conditions, influencing mood and behavior."
            ],
        }, "UF_R": {
            "id": 31,
            "tract": "uncinate fasciculus",
            "side": "right",
            "type": "association",
            "functional_significance": [
                "Supports the integration of emotional content with visual and spatial information.",
                "Involved in the recognition and processing of non-verbal emotional cues, such as facial expressions.",
                "Plays a role in the formation and retrieval of emotionally charged memories.",
                "Contributes to intuitive decision-making by linking emotional and spatial information.",
                "Facilitates social and emotional processing, supporting empathy and social awareness."
            ],
            "structural_characteristics": [
                "Connects the orbitofrontal cortex with the anterior temporal regions, supporting emotional and social functions.",
                "The right UF is particularly involved in processing and integrating emotional and non-verbal cues.",
                "Structural design facilitates the efficient processing of complex emotional and spatial information.",
                "Myelination patterns support quick relay of emotional signals, essential for rapid emotional responses.",
                "The integrity of the UF is vital for the effective interpretation of social and emotional stimuli."
            ],
            "vulnerability_to_disease": [
                "Damage can result in impaired recognition of emotional expressions and social cues.",
                "Disruptions can lead to changes in social behavior and emotional regulation.",
                "Degenerative conditions affecting the right UF may impair emotional memory and empathy.",
                "Traumatic injury to the right UF can affect social cognition and emotional processing.",
                "Psychiatric disorders may involve alterations in the UF, affecting emotional and social functioning."
            ],
        }
    }






    LABELS = {value["id"]: key for key, value in TRACT_LIST.items()}# Diccionario id -> Etiqueta
    caption_templates = [
            "Exploring the {type} tract of {tract} on the {side}, pivotal for {functional_significance}, characterized by {structural_characteristics}, with known {vulnerability_to_disease}.",
            "The {type} {tract} on the {side} side: A key player in {functional_significance}, structured with {structural_characteristics}, and susceptible to {vulnerability_to_disease}.",
            "Insights into the {tract}'s {type} fibers on the {side}: Essential for {functional_significance}, featuring {structural_characteristics}, and vulnerable to {vulnerability_to_disease}.",
            "{type} fibers of the {tract} on the {side} side underscore {functional_significance}, demonstrate {structural_characteristics}, and face threats from {vulnerability_to_disease}.",
            "Highlighting the {type} tract, {tract}, on the {side} side: Central to {functional_significance}, with {structural_characteristics}, and at risk of {vulnerability_to_disease}.",
            "The {tract}'s {type} fibers on the {side}: Integral for {functional_significance}, endowed with {structural_characteristics}, yet at risk due to {vulnerability_to_disease}.",
            "Delving into the {tract} on the {side} side: A {type} fiber crucial for {functional_significance}, defined by {structural_characteristics}, with a susceptibility to {vulnerability_to_disease}.",
            "A closer look at the {type} {tract} on the {side}, a cornerstone of {functional_significance}, built on {structural_characteristics}, with exposure to {vulnerability_to_disease}.",
            "Understanding the {type} {tract} on the {side}: Fundamental for {functional_significance}, based on {structural_characteristics}, with potential for {vulnerability_to_disease}.",
            "The {tract} on the {side}, a {type} tract: Vital for {functional_significance}, composed of {structural_characteristics}, and prone to {vulnerability_to_disease}.",
            "Navigating the {tract}'s {type} fibers on the {side}: Key to {functional_significance}, showcasing {structural_characteristics}, while facing {vulnerability_to_disease}.",
            "{type} fiber dynamics in the {tract} on the {side}: Driving {functional_significance}, supported by {structural_characteristics}, with concerns over {vulnerability_to_disease}.",
            "The role of the {type} {tract} on the {side} in {functional_significance}, its structure defined by {structural_characteristics}, and its challenges with {vulnerability_to_disease}.",
            "Spotlight on the {tract}'s {type} aspect on the {side}: A linchpin in {functional_significance}, with a basis in {structural_characteristics}, amid {vulnerability_to_disease}.",
            "Charting the course of the {type} {tract} on the {side}: Crucial for {functional_significance}, with a foundation of {structural_characteristics}, challenged by {vulnerability_to_disease}.",
            "Dissecting the {type} {tract} on the {side}: Central to {functional_significance}, with distinct {structural_characteristics}, and under threat from {vulnerability_to_disease}.",
            "The {type} fibers within the {tract} on the {side}: Pillars of {functional_significance}, with unique {structural_characteristics}, and susceptible to {vulnerability_to_disease}.",
            "Unveiling the {type} {tract} on the {side}: A conduit for {functional_significance}, designed with {structural_characteristics}, yet vulnerable to {vulnerability_to_disease}.",
            "The architecture of the {type} {tract} on the {side}: Scaffolding for {functional_significance}, erected on {structural_characteristics}, with a risk of {vulnerability_to_disease}.",
            "Deciphering the {type} {tract} on the {side}: Essential for {functional_significance}, with {structural_characteristics}, and facing risks of {vulnerability_to_disease}.",
            "The {tract}'s {type} path on the {side}: Spearheading {functional_significance}, built upon {structural_characteristics}, with vulnerabilities to {vulnerability_to_disease}.",
            "Insight into the {type} {tract} on the {side}: Spearheads {functional_significance}, relying on {structural_characteristics}, amidst threats from {vulnerability_to_disease}.",
            "Showcasing the {type} {tract} on the {side}: A crucible for {functional_significance}, framed by {structural_characteristics}, with exposure to {vulnerability_to_disease}.",
            "The essence of the {type} {tract} on the {side}: Harnessing {functional_significance}, through {structural_characteristics}, with a nod to {vulnerability_to_disease}.",
            "Evaluating the {type} {tract} on the {side}: A nexus for {functional_significance}, supported by {structural_characteristics}, with an eye on {vulnerability_to_disease}.",
            "Delineating the {side} {type} {tract}: A beacon for {functional_significance}, woven with {structural_characteristics}, under the shadow of {vulnerability_to_disease}.",
            "The {tract} on the {side}: A {type} network pivotal in {functional_significance}, structured via {structural_characteristics}, with an Achilles' heel of {vulnerability_to_disease}.",
            "A deep dive into the {type} {tract} on the {side}: Orchestrating {functional_significance}, through the lens of {structural_characteristics}, amidst battles with {vulnerability_to_disease}.",
            "Unraveling the {type} {tract} on the {side}: A cornerstone in {functional_significance}, pieced together by {structural_characteristics}, with a vulnerability to {vulnerability_to_disease}.",
            "Illuminating the {type} {tract} on the {side}: A catalyst for {functional_significance}, held together by {structural_characteristics}, yet dancing with {vulnerability_to_disease}.",
            "Navigating the nuances of the {type} {tract} on the {side}: Engineered for {functional_significance}, detailed by {structural_characteristics}, in the face of {vulnerability_to_disease}.",
            "The intricate web of the {type} {tract} on the {side}: A symphony of {functional_significance}, composed of {structural_characteristics}, with whispers of {vulnerability_to_disease}.",
            "The {tract} on the {side}: Where {type} fibers blend {functional_significance} with {structural_characteristics}, amidst the specter of {vulnerability_to_disease}.",
            "A glimpse into the {type} {tract} on the {side}: Harmonizing {functional_significance} with {structural_characteristics}, against the backdrop of {vulnerability_to_disease}.",
            "The {tract}'s {type} ensemble on the {side}: An odyssey of {functional_significance}, mapped through {structural_characteristics}, with a narrative of {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}: A testament to {functional_significance}, sculpted by {structural_characteristics}, with tales of {vulnerability_to_disease}.",
            "Decoding the {type} {tract} on the {side}: A saga of {functional_significance}, etched with {structural_characteristics}, shadowed by {vulnerability_to_disease}.",
            "The {tract} on the {side}: A {type} odyssey defined by {functional_significance}, articulated through {structural_characteristics}, with challenges of {vulnerability_to_disease}.",
            "Embarking on the {type} {tract} journey on the {side}: Driven by {functional_significance}, carved out of {structural_characteristics}, with hurdles of {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}: A crucible of {functional_significance}, molded by {structural_characteristics}, with the trials of {vulnerability_to_disease}.",
            "The {tract}'s {type} legacy on the {side}: Stitched with {functional_significance}, through the fabric of {structural_characteristics}, bracing against {vulnerability_to_disease}.",
            "Exploring the {type} {tract} on the {side}: A quilt of {functional_significance}, with patches of {structural_characteristics}, and threads of {vulnerability_to_disease}.",
            "The {tract} on the {side}: A {type} tapestry rich in {functional_significance}, woven with {structural_characteristics}, tinged with {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}: A mosaic of {functional_significance}, pieced together by {structural_characteristics}, with cracks of {vulnerability_to_disease}.",
            "Navigating the {type} {tract} on the {side}: A journey through {functional_significance}, charted by {structural_characteristics}, navigating {vulnerability_to_disease}.",
            "The {tract} on the {side}: A {type} vista overlooking {functional_significance}, painted with {structural_characteristics}, under clouds of {vulnerability_to_disease}.",
            "Unlocking the {type} {tract} on the {side}: A vault of {functional_significance}, secured by {structural_characteristics}, under siege by {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}: A beacon across {functional_significance}, shining through {structural_characteristics}, amid storms of {vulnerability_to_disease}.",
            "Journeying through the {type} {tract} on the {side}: A pathway lit by {functional_significance}, paved with {structural_characteristics}, weaving through {vulnerability_to_disease}.",
            "The {tract} on the {side}: A {type} symposium of {functional_significance}, curated with {structural_characteristics}, amidst the risks of {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}: A dialogue between {functional_significance} and {structural_characteristics}, with an undercurrent of {vulnerability_to_disease}.",
            "Unraveling the {type} {tract} on the {side}: Threads of {functional_significance} intertwined with {structural_characteristics}, frayed by {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}: A narrative woven from {functional_significance}, bound by {structural_characteristics}, and shadowed by {vulnerability_to_disease}.",
            "Charting the {type} {tract} on the {side}: A map of {functional_significance}, drawn with {structural_characteristics}, bordered by {vulnerability_to_disease}.",
            "The {tract} on the {side}: A {type} cornerstone anchoring {functional_significance}, chiseled from {structural_characteristics}, weathering {vulnerability_to_disease}.",
            "Amidst {vulnerability_to_disease}, the {type} {tract} on the {side} side stands out, marked by {structural_characteristics} and pivotal in {functional_significance}.",
            "Where {functional_significance} meets {structural_characteristics}, the {type} {tract} on the {side} navigates through {vulnerability_to_disease}.",
            "{vulnerability_to_disease} shadows the {type} {tract} on the {side}, a structure of {structural_characteristics} essential for {functional_significance}.",
            "In the realm of {functional_significance}, the {type} {tract} on the {side} emerges, built on {structural_characteristics}, yet tested by {vulnerability_to_disease}.",
            "Challenged by {vulnerability_to_disease}, the {type} {tract} on the {side} thrives, an epitome of {structural_characteristics} serving {functional_significance}.",
            "{structural_characteristics} define the {type} {tract} on the {side}, a beacon for {functional_significance} amidst {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}, driven by {functional_significance}, showcases {structural_characteristics} while confronting {vulnerability_to_disease}.",
            "{functional_significance} is the hallmark of the {type} {tract} on the {side}, crafted from {structural_characteristics} and facing {vulnerability_to_disease}.",
            "Against the backdrop of {vulnerability_to_disease}, the {type} {tract} on the {side} excels in {functional_significance}, thanks to its {structural_characteristics}.",
            "The {type} {tract} on the {side}, a testament to {structural_characteristics}, champions {functional_significance} amid {vulnerability_to_disease}.",
            "Forged with {structural_characteristics}, the {type} {tract} on the {side} propels {functional_significance}, even as it grapples with {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}, sculpted by {structural_characteristics}, enriches {functional_significance} against {vulnerability_to_disease}.",
            "Bearing {structural_characteristics}, the {type} {tract} on the {side} underpins {functional_significance}, resilient to {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}, with its {structural_characteristics}, is a linchpin in {functional_significance}, enduring {vulnerability_to_disease}.",
            "Rooted in {structural_characteristics}, the {type} {tract} on the {side} elevates {functional_significance}, despite {vulnerability_to_disease}.",
            "Empowered by {structural_characteristics}, the {type} {tract} on the {side} fuels {functional_significance}, transcending {vulnerability_to_disease}.",
            "{vulnerability_to_disease} tests the {type} {tract} on the {side}, a construct of {structural_characteristics} and a vessel for {functional_significance}.",
            "The {type} {tract} on the {side}: a synthesis of {structural_characteristics}, serving {functional_significance}, amidst trials of {vulnerability_to_disease}.",
            "With {structural_characteristics} at its core, the {type} {tract} on the {side} epitomizes {functional_significance}, braving {vulnerability_to_disease}.",
            "A nexus of {structural_characteristics}, the {type} {tract} on the {side} radiates {functional_significance}, undeterred by {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}, a paragon of {structural_characteristics}, emboldens {functional_significance}, despite {vulnerability_to_disease}.",
            "Anchored in {structural_characteristics}, the {type} {tract} on the {side} is a crucible for {functional_significance}, amidst {vulnerability_to_disease}.",
            "{structural_characteristics} are the foundation of the {type} {tract} on the {side}, powering {functional_significance}, amidst {vulnerability_to_disease}.",
            "A citadel of {structural_characteristics}, the {type} {tract} on the {side} is pivotal for {functional_significance}, despite {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}, framed by {structural_characteristics}, is a bastion of {functional_significance}, facing {vulnerability_to_disease}.",
            "At the intersection of {structural_characteristics} and {functional_significance}, the {type} {tract} on the {side} confronts {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}, a convergence of {structural_characteristics}, propels {functional_significance}, against the tide of {vulnerability_to_disease}.",
            "Imbued with {structural_characteristics}, the {type} {tract} on the {side} amplifies {functional_significance}, navigating {vulnerability_to_disease}.",
            "The {type} {tract} on the {side}, an embodiment of {structural_characteristics}, fortifies {functional_significance}, in defiance of {vulnerability_to_disease}.",
            "With {structural_characteristics} as its bedrock, the {type} {tract} on the {side} is instrumental in {functional_significance}, amidst the waves of {vulnerability_to_disease}."

        ]
    
    
    graph_labels = torch.tensor([graph.y.item() for graph in batch], dtype=torch.long)
    
    # 1 positive pair -> text corresponds to graph
    # 0 negative pair -> text does not correspond to graph
    type_of_pair = (torch.rand(len(graph_labels)) < CFG.pos_neg_pair_ratio).long()
    

    
    text_labels = graph_labels.clone()
    for pos, value in enumerate(type_of_pair):
        if value == 0:
            text_labels[pos] = torch.randint(0, 32, (1,)).item()

    # Recuperar y tokenizar todos los captions necesarios en una sola llamada
    # captions = [random.choice(caption_templates).format(**TRACT_LIST[LABELS[label.item()]]) for label in text_labels]
    captions = []
    for label in text_labels:
        tract = TRACT_LIST[LABELS[label.item()]]
        input_data = {
            "type": tract["type"],
            "tract": tract["tract"],
            "side": tract["side"],
            "functional_significance": random.choice(tract["functional_significance"]),
            "structural_characteristics": random.choice(tract["structural_characteristics"]),
            "vulnerability_to_disease": random.choice(tract["vulnerability_to_disease"])
        }
        captions.append(random.choice(caption_templates).format(**input_data))
    tokenized_texts_batch = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    
    # Devolver el lote de grafos, los textos tokenizados, los graph_labels, los pos_neg_labels y los type_of_pair
    return GeoBatch.from_data_list(batch), {'input_ids': tokenized_texts_batch['input_ids'], 'attention_mask': tokenized_texts_batch['attention_mask']}, graph_labels, text_labels, type_of_pair # 1 positive pair -> text corresponds to graph, 0 negative pair -> text does not correspond to graph
            

    

    