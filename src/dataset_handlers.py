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
                'side' : 'left',
                'type': 'association'
            },'AF_R': {
                'id': 1, 
                'tract': 'arcuate fasciculus',
                'side' : 'right',
                'type': 'association'
            },'CC_Fr_1': {
                'id': 2, 
                'tract': 'corpus callosum, frontal lobe',
                'side' : 'most anterior part of the frontal lobe', 
                'type': 'commissural'
            },'CC_Fr_2': {
                'id': 3, 
                'tract': 'corpus callosum, frontal lobe',
                'side' : 'most posterior part of the frontal lobe',
                'type': 'commissural'
            },'CC_Oc': {
                'id': 4, 
                'tract': 'corpus callosum, occipital lobe',
                'side' : 'central',
                'type': 'commissural'
            },'CC_Pa': {
                'id': 5, 
                'tract': 'corpus callosum, parietal lobe',
                'side' : 'central',
                'type': 'commissural'
            },'CC_Pr_Po': {
                'id': 6, 
                'tract': 'corpus callosum, pre/post central gyri',
                'side' : 'central',
                'type': 'commissural'
            },'CG_L': {
                'id': 7, 
                'tract': 'cingulum',
                'side' : 'left',
                'type': 'association'
            },'CG_R': {
                'id': 8,
                'tract': 'cingulum',
                'side' : 'right',
                'type': 'association'
            },'FAT_L': {
                'id': 9,
                'tract': 'frontal aslant tract',
                'side' : 'left',
                'type': 'association'
            },'FAT_R': {
                'id': 10,
                'tract': 'frontal aslant tract',
                'side' : 'right',
                'type': 'association'
            },'FPT_L': {
                'id': 11,
                'tract': 'fronto-pontine tract',
                'side' : 'left',
                'type': 'association' 
            },'FPT_R': {
                'id': 12, 
                'tract': 'fronto-pontine tract',
                'side' : 'right',
                'type': 'association'
            }, 'FX_L': {
                'id': 13,
                'tract': 'fornix',
                'side' : 'left',
                'type': 'commissural'
            },'FX_R': {
                'id': 14,
                'tract': 'fornix',
                'side' : 'right',
                'type': 'commissural'
            },'IFOF_L': {
                'id': 15,
                'tract': 'inferior fronto-occipital fasciculus',
                'side' : 'left',
                'type': 'association'
            },'IFOF_R': {
                'id': 16,
                'tract': 'inferior fronto-occipital fasciculus',
                'side' : 'right',
                'type': 'association'
            },'ILF_L': {
                'id': 17,
                'tract': 'inferior longitudinal fasciculus',
                'side' : 'left',
                'type': 'association'
            },'ILF_R': {
                'id': 18,
                'tract': 'inferior longitudinal fasciculus',
                'side' : 'right',
                'type': 'association'
            },'MCP': {
                'id': 19,
                'tract': 'middle cerebellar peduncle',
                'side' : 'central',
                'type': 'commissural'
            },'MdLF_L': {
                'id': 20,
                'tract': 'middle longitudinal fasciculus',
                'side' : 'left',
                'type': 'association'
            },'MdLF_R': {
                'id': 21,
                'tract': 'middle longitudinal fasciculus',
                'side' : 'right',
                'type': 'association'
            },'OR_ML_L': {
                'id': 22,
                'tract': 'optic radiation, Meyer loop',
                'side' : 'left',
                'type': 'projection'
            },'OR_ML_R': {
                'id': 23,
                'tract': 'optic radiation, Meyer loop',
                'side' : 'right',
                'type': 'projection'
            },'POPT_L': {
                'id': 24,
                'tract': 'pontine crossing tract',
                'side' : 'left',
                'type': 'commissural'
            },'POPT_R': {
                'id': 25, 
                'tract': 'pontine crossing tract',
                'side' : 'right',
                'type': 'commissural'
            },'PYT_L': {
                'id': 26,
                'tract': 'pyramidal tract',
                'side' : 'left',
                'type': 'projection' 
            },'PYT_R': {
                'id': 27,
                'tract': 'pyramidal tract',
                'side' : 'right',
                'type': 'projection' 
            },'SLF_L': {
                'id': 28,
                'tract': 'superior longitudinal fasciculus',
                'side' : 'left',
                'type': 'association' 
            },'SLF_R': {
                'id': 29,
                'tract': 'superior longitudinal fasciculus',
                'side' : 'right',
                'type': 'association' 
            },'UF_L': {
                'id': 30,
                'tract': 'uncinate fasciculus',
                'side' : 'left',
                'type': 'association'
            },'UF_R': {
                'id': 31,
                'tract': 'uncinate fasciculus',
                'side' : 'right',
                'type': 'association'
            }
        }
    
        # Diccionario igual pero cambiando el valor de id por la key Ej: {'0': 'AF_L', '1': 'AF_R', ...}
        self.LABELS = {value["id"]: key for key, value in self.TRACT_LIST.items()}
        print(self.LABELS)
    

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
    ds_handler = Tractoinferno_handler(tractoinferno_path = "/app/dataset/derivatives", scope = "testset")
    # ds_handler = Tractoinferno_handler(tractoinferno_path = r"C:\Users\pablo\Documents\Datasets\ds003900-download\derivatives", scope = "testset")
    
    paths = ds_handler.get_data()
    print(paths, len(paths))