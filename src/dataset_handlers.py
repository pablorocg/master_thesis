from abc import ABC, abstractmethod
import pathlib2 as pathlib

class Dataset_handler(ABC):
    """
    Clase abstracta para el manejo de las rutas de un dataset.

    Args:

    Attributes:
        subjects (list[pathlib.Path]): Lista de las rutas de los sujetos del dataset.
        scope (str): Scope del dataset. Puede ser 'trainset', 'validset', 'testset' o 'all'.
    """
    def __init__(self, ds_path:str, scope:str):
        """
        Esta función debe devolver una lista con las rutas de los sujetos
        dentro del scope del dataset.

        Args:
            scope (str): Scope del dataset. Puede ser 'trainset', 'validset', 'testset' o 'all'.
        """
        self.ds_path = ds_path
        self.subjects = self._get_subjects()
        self.scope = scope
        pass

    @abstractmethod
    def _get_subjects(self) -> list[pathlib.Path]:
        """
        Esta función debe devolver una lista con las rutas de los sujetos 
        dentro del scope del dataset.
        """
        pass

    @abstractmethod
    def get_tract_paths_from_suj(self, subject:pathlib.Path, tract:str = "all"):
        """
        Esta función debe devolver una lista con las rutas de los tractos de un sujeto.
        """
        pass

    @abstractmethod
    def get_anat_from_subject(self, subject:pathlib.Path):
        """
        Esta función debe devolver la ruta de la imagen anatómica de un sujeto (T1, T2, ...).
        """
        pass

    @abstractmethod
    def get_data_from_subject(self, subject:pathlib.Path):
        """
        Esta función debe devolver un diccionario con la información de un sujeto.
        """
        pass

    def __len__(self):
        """
        Esta función devuelve el número de sujetos incluidos en el scope del dataset.
        """
        return len(self.subjects)
    

#===========================TRACTOINFERNO HANDLER CLASS===========================#
class Tractoinferno_handler(Dataset_handler):
    """
    Clase para la generación del dataset tractoinferno. 
    Para poder utilizar el dataset Tractoinferno es necesario descargarlo previamente desde https://openneuro.org/datasets/ds003900/versions/1.1.1
    y colocar la ruta de la subcarpeta 'derivatives' en el parámetro 'tractoinferno_path' de la clase Tractoinferno.
    """

    def __init__(self, 
                 tractoinferno_path:str, 
                 scope: str = "all"):

        # crear un path con pathlib2 y comprobar que existe
        if pathlib.Path(tractoinferno_path).exists():
            self.tractoinferno_path = pathlib.Path(tractoinferno_path)
        else:
            raise ValueError("La ruta al dataset tractoinferno no existe.")
        
        # Comprobar que el scope es correcto
        if scope not in ["trainset", "validset", "testset", "all"]:
            raise ValueError("El scope debe ser 'trainset', 'validset', 'testset' o 'all'.")
        else:
            self.scope = scope # Scope del dataset
            self.subjects = self._get_subjects()# Lista de rutas de los sujetos del dataset

        # Lista de tractos
        self.TRACT_LIST = {
            'AF_L': {
                'id': 0,
                'tract': 'arcuate fasciculus',
                'localization': 'left hemisphere',
                'description': 'The arcuate fasciculus is a major white matter tract connecting frontal and temporal lobes. It is involved in language and auditory processing.'
            },'AF_R': {
                'id': 1, 
                'name': 'Right part of the arcuate fasciculus',
                'description': 'The arcuate fasciculus is a major white matter tract connecting frontal and temporal lobes. It is involved in language and auditory processing.'
            },'CC_Fr_1': {
                'id': 2, 
                'name': 'Corpus callosum, Frontal lobe (most anterior part)',
                'description': 'The corpus callosum is a major white matter tract connecting the two hemispheres. It is involved in many functions including interhemispheric transfer of motor, sensory, and cognitive information.'
            },'CC_Fr_2': {
                'id': 3, 
                'name': 'Corpus callosum, Frontal lobe (most posterior part)',
                'description': 'The corpus callosum is a major white matter tract connecting the two hemispheres. It is involved in many functions including interhemispheric transfer of motor, sensory, and cognitive information.'
            },'CC_Oc': {
                'id': 4, 
                'name': 'Corpus callosum, Occipital lobe',
                'description': 'The corpus callosum is a major white matter tract connecting the two hemispheres. It is involved in many functions including interhemispheric transfer of motor, sensory, and cognitive information.'
            },'CC_Pa': {
                'id': 5, 
                'name': 'Corpus callosum, Parietal lobe',
                'description': 'The corpus callosum is a major white matter tract connecting the two hemispheres. It is involved in many functions including interhemispheric transfer of motor, sensory, and cognitive information.'
            },'CC_Pr_Po': {
                'id': 6, 
                'name': 'Corpus callosum, Pre/Post central gyri',
                'description': 'The corpus callosum is a major white matter tract connecting the two hemispheres. It is involved in many functions including interhemispheric transfer of motor, sensory, and cognitive information.'
            },'CG_L': {
                'id': 7, 
                'name': 'Left part of the cingulum',
                'description': 'The cingulum is a major white matter tract connecting frontal and parietal lobes. It is involved in many functions including emotion, learning, and memory.'
            },'CG_R': {
                'id': 8, 
                'name': 'Right part of the cingulum',
                'description': 'The cingulum is a major white matter tract connecting frontal and parietal lobes. It is involved in many functions including emotion, learning, and memory.'
            },'FAT_L': {
                'id': 9, 
                'name': 'Left part of the frontal aslant tract',
                'description': 'The frontal aslant tract is a major white matter tract connecting frontal and prefrontal lobes. It is involved in language processing.'
            },'FAT_R': {
                'id': 10, 
                'name': 'Right part of the frontal aslant tract',
                'description': 'The frontal aslant tract is a major white matter tract connecting frontal and prefrontal lobes. It is involved in language processing.'
            },'FPT_L': {
                'id': 11, 
                'name': 'Left part of the frontopontine tract',
                'description': 'The frontopontine tract is a major white matter tract connecting frontal and pontine lobes. It is involved in motor control.'
            },'FPT_R': {
                'id': 12, 
                'name': 'Right part of the frontopontine tract',
                'description': 'The frontopontine tract is a major white matter tract connecting frontal and pontine lobes. It is involved in motor control.'
            }, 'FX_L': {
                'id': 13,
                'name': 'Left part of the fornix',
                'description': 'The fornix is a major white matter tract connecting hippocampus and mammillary bodies. It is involved in memory processing.'
            },'FX_R': {
                'id': 14,
                'name': 'Right part of the fornix',
                'description': 'The fornix is a major white matter tract connecting hippocampus and mammillary bodies. It is involved in memory processing.'
            },'IFOF_L': {
                'id': 15, 
                'name': 'Left part of the inferior fronto-occipital fasciculus',
                'description': 'The inferior fronto-occipital fasciculus is a major white matter tract connecting frontal and occipital lobes. It is involved in language processing.'
            },'IFOF_R': {
                'id': 16, 
                'name': 'Right part of the inferior fronto-occipital fasciculus',
                'description': 'The inferior fronto-occipital fasciculus is a major white matter tract connecting frontal and occipital lobes. It is involved in language processing.'
            },'ILF_L': {
                'id': 17, 
                'name': 'Left part of the inferior longitudinal fasciculus',
                'description': 'The inferior longitudinal fasciculus is a major white matter tract connecting temporal and occipital lobes. It is involved in language processing.'
            },'ILF_R': {
                'id': 18, 
                'name': 'Right part of the inferior longitudinal fasciculus',
                'description': 'The inferior longitudinal fasciculus is a major white matter tract connecting temporal and occipital lobes. It is involved in language processing.'
            },'MCP': {
                'id': 19, 
                'name': 'Middle cerebellar peduncle',
                'description': 'The middle cerebellar peduncle is a major white matter tract connecting cerebellum and pons. It is involved in motor control.'   
            },'MdLF_L': {
                'id': 20, 
                'name': 'Left part of the middle longitudinal fascicle',
                'description': 'The middle longitudinal fascicle is a major white matter tract connecting temporal and parietal lobes. It is involved in language processing.'
            },'MdLF_R': {
                'id': 21, 
                'name': 'Right part of the middle longitudinal fascicle',
                'description': 'The middle longitudinal fascicle is a major white matter tract connecting temporal and parietal lobes. It is involved in language processing.'
            },'OR_ML_L': {
                'id': 22, 
                'name': 'Left part of the optic radiation and Meyer’s loop',
                'description': 'The optic radiation is a major white matter tract connecting lateral geniculate nucleus and visual cortex. It is involved in visual processing.'
            },'OR_ML_R': {
                'id': 23, 
                'name': 'Right part of the optic radiation and Meyer’s loop',
                'description': 'The optic radiation is a major white matter tract connecting lateral geniculate nucleus and visual cortex. It is involved in visual processing.'
            },'POPT_L': {
                'id': 24, 
                'name': 'Left part of the parieto-occipito pontine tract',
                'description': 'The parieto-occipito pontine tract is a major white matter tract connecting parietal and occipital lobes. It is involved in motor control.'
            },'POPT_R': {
                'id': 25, 
                'name': 'Right part of the parieto-occipito pontine tract',
                'description': 'The parieto-occipito pontine tract is a major white matter tract connecting parietal and occipital lobes. It is involved in motor control.'
            },'PYT_L': {
                'id': 26, 
                'name': 'Left part of the pyramidal tract',
                'description': 'The pyramidal tract is a major white matter tract connecting motor cortex and spinal cord. It is involved in motor control.'
            },'PYT_R': {
                'id': 27, 
                'name': 'Right part of the pyramidal tract',
                'description': 'The pyramidal tract is a major white matter tract connecting motor cortex and spinal cord. It is involved in motor control.'
            },'SLF_L': {
                'id': 28, 
                'name': 'Left part of the superior longitudinal fasciculus',
                'description': 'The superior longitudinal fasciculus is a major white matter tract connecting frontal and parietal lobes. It is involved in many functions including language processing.'
            },'SLF_R': {
                'id': 29, 
                'name': 'Right part of the superior longitudinal fasciculus',
                'description': 'The superior longitudinal fasciculus is a major white matter tract connecting frontal and parietal lobes. It is involved in many functions including language processing.'
            },'UF_L': {
                'id': 30, 
                'name': 'Left part of the uncinate fasciculus',
                'description': 'The uncinate fasciculus is a major white matter tract connecting frontal and temporal lobes. It is involved in language processing.'
            },'UF_R': {
                'id': 31, 
                'name': 'Right part of the uncinate fasciculus',
                'description': 'The uncinate fasciculus is a major white matter tract connecting frontal and temporal lobes. It is involved in language processing.'
            }
            }

    def _get_subjects(self) -> list[pathlib.Path]:
        """
        Function to get the subjects of the dataset.

        Returns:
            list[pathlib.Path]: List of the subjects of the dataset.
        """
        if self.scope == "all":
            # Devuelve la ruta de todas las subcarpetas de las carpetas trainset, validationset y testset
            return [path for path in self.tractoinferno_path.glob("*/*") if path.is_dir()]
        elif self.scope in ["trainset", "validset", "testset"]:
            # Devuelve la ruta de todas las subcarpetas de la carpeta trainset, validationset o testset
            return [path for path in self.tractoinferno_path.joinpath(self.scope).glob("*") if path.is_dir()]
        

    def get_tract_paths_from_suj(self, 
                                 subject:pathlib.Path, 
                                 tract:str = "all", 
                                 extension:str = "trk"):
        """
        Function to get the paths of the tracts of a subject.

        Args:
            subject (str): Name of the subject.
            tract (str): Name of the tract.

        Returns:
            list[str]: List of the paths of the tracts of a subject.
        """
        if tract not in self.TRACT_LIST.keys() and tract != "all":
            raise ValueError("El tracto no existe.")
        
        if tract == "all":
            # Devuelve la ruta de todos los tractos de un sujeto
            return [path for path in subject.joinpath("tractography").glob(f"*.{extension}")]
        else:
            # Devuelve la ruta de un tracto de un sujeto
            return [path for path in (subject.joinpath("tractography").glob(f"*{tract}*.{extension}"))]
            
                

    def get_anat_from_subject(self, subject:pathlib.Path):
        """
        Devuelve los streamlines y la etiqueta de un sujeto y un tracto.

        Args:
            subject (str): Nombre del sujeto.
            tract (str): Nombre del tracto.

        Returns:
            tuple[list[np.ndarray], str]: Lista de arrays con los streamlines y la etiqueta del tracto.
        """
        # Comprobar que el sujeto existe
        try:
            return [path for path in subject.joinpath("anat").glob(f"*.nii.gz")][0]
        except KeyError:
            raise ValueError("El sujeto no existe.")


    def get_data_from_subject(self, subject:pathlib.Path, extension:str = "trk") -> dict:

        anat = self.get_anat_from_subject(subject) # Devuelve la imagen T1w
        tracts = self.get_tract_paths_from_suj(subject, extension = extension) # Devuelve la lista de tractos de un sujeto
        return {"subject": subject.name, 
                "subject_split":subject.parent.name, 
                "T1w": anat, 
                "tracts": tracts} 
    
    def get_label_from_tract(self, tract:str) -> str:
        """
        Devuelve la etiqueta de un tracto.

        Args:
            tract (str): Nombre del tracto.

        Returns:
            str: Etiqueta del tracto.
        """
        return self.TRACT_LIST[tract]["id"]
    
    def get_tract_from_label(self, label:int) -> str:
        """
        Devuelve el nombre de un tracto a partir de su etiqueta.

        Args:
            label (int): Etiqueta del tracto.

        Returns:
            str: Nombre del tracto.
        """
        for tract in self.TRACT_LIST.keys():
            if self.TRACT_LIST[tract]["id"] == label:
                return tract
    
    def get_data(self) -> list[dict]:
        """ 
        Devuelve una lista con los datos de todos los sujetos del dataset en .

        lista[dict{Path, list[Path]}]

        salida -> [sujeto1{anat, [tracto1, tracto2, ...]}, sujeto2{anat, [tracto1, tracto2, ...]}, ...]
        """
        return [self.get_data_from_subject(subject) for subject in self.subjects]
    


if __name__ == "__main__":
    # ds_handler = Tractoinferno_handler(tractoinferno_path = r"C:\Users\pablo\GitHub\tfm_prg\tractoinferno_preprocessed_mni", 
    #                                    scope = "testset")
    ds_handler = Tractoinferno_handler(tractoinferno_path = r"C:\Users\pablo\Documents\Datasets\ds003900-download\derivatives", 
                                       scope = "testset")
    
    paths = ds_handler.get_data()
    print(paths, len(paths))