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
                 path:str, 
                 scope: str = "all"):

        # crear un path con pathlib2 y comprobar que existe
        if pathlib.Path(path).exists():
            self.tractoinferno_path = pathlib.Path(path)
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
    

class HCP_handler:
    def __init__(self, path:str, scope:str):
        # HCP FOLDS
        self.fold1 = ['992774', '991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367', '959574', '958976', '957974', '951457', '932554', '930449', '922854', '917255', '912447', '910241', '907656', '904044']
        self.fold2 = ['901442', '901139', '901038', '899885', '898176', '896879', '896778', '894673', '889579', '887373', '877269', '877168', '872764', '872158', '871964', '871762', '865363', '861456', '859671', '857263', '856766']
        self.fold3 = ['849971', '845458', '837964', '837560', '833249', '833148', '826454', '826353', '816653', '814649', '802844', '792766', '792564', '789373', '786569', '784565', '782561', '779370', '771354', '770352', '765056']
        self.fold4 = ['761957', '759869', '756055', '753251', '751348', '749361', '748662', '748258', '742549', '734045', '732243', '729557', '729254', '715647', '715041', '709551', '705341', '704238', '702133', '695768', '690152']
        self.fold5 = ['687163', '685058', '683256', '680957', '679568', '677968', '673455', '672756', '665254', '654754', '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671', '599469']

        self.trainset = self.fold1 + self.fold2 + self.fold3 
        self.validset = self.fold4
        self.testset = self.fold5


        # crear un path con pathlib2 y comprobar que existe
        if pathlib.Path(path).exists():
            self.hcp_path = pathlib.Path(path)
        else:
            raise ValueError("La ruta al dataset HCP 105 no existe.")
        
        # Comprobar que el scope es correcto
        if scope not in ["trainset", "validset", "testset", "all"]:
            raise ValueError("El scope debe ser 'trainset', 'validset', 'testset' o 'all'.")
        else:
            self.scope = scope
            self.subjects = self._get_subjects()

        # Lista de tractos
        self.TRACT_LIST = {
            'AF_left': {'id': 0, 'tract':'arcuate fasciculus', 'side':'left'},
            'AF_right': {'id': 1, 'tract':'arcuate fasciculus', 'side':'right'},
            'ATR_left': {'id': 2, 'tract':'anterior thalamic radiation', 'side':'left'},
            'ATR_right': {'id': 3, 'tract':'anterior thalamic radiation', 'side':'right'},
            'CA': {'id': 4, 'tract':'commissure anterior', 'side':'NA'},
            'CC_1': {'id': 5, 'tract':'corpus callosum', 'side':'rostrum'},
            'CC_2': {'id': 6, 'tract':'corpus callosum', 'side':'genu'},
            'CC_3': {'id': 7, 'tract':'corpus callosum', 'side':'rostral body (premotor)'},
            'CC_4': {'id': 8, 'tract':'corpus callosum', 'side':'anterior midbody (primary motor)'},
            'CC_5': {'id': 9, 'tract':'corpus callosum', 'side':'posterior midbody (primary somatosensory)'},
            'CC_6': {'id': 10, 'tract':'corpus callosum', 'side':'isthmus'},
            'CC_7': {'id': 11, 'tract':'corpus callosum', 'side':'splenium'},
            'CC': {'id': 12, 'tract':'corpus callosum', 'side':'all'},
            'CG_left': {'id': 13, 'tract':'cingulum', 'side':'left'},
            'CG_right': {'id': 14, 'tract':'cingulum', 'side':'right'},
            'CST_left': {'id': 15, 'tract':'corticospinal tract', 'side':'left'},
            'CST_right': {'id': 16, 'tract':'corticospinal tract', 'side':'right'},
            'MLF_left': {'id': 17, 'tract':'middle longitudinal fascicle', 'side':'left'},
            'MLF_right': {'id': 18, 'tract':'middle longitudinal fascicle', 'side':'right'},
            'FPT_left': {'id': 19, 'tract':'fronto-pontine tract', 'side':'left'},
            'FPT_right': {'id': 20, 'tract':'fronto-pontine tract', 'side':'right'},
            'FX_left': {'id': 21, 'tract':'fornix', 'side':'left'},
            'FX_right': {'id': 22, 'tract':'fornix', 'side':'right'},
            'ICP_left': {'id': 23, 'tract':'inferior cerebellar peduncle', 'side':'left'},
            'ICP_right': {'id': 24, 'tract':'inferior cerebellar peduncle', 'side':'right'},
            'IFO_left': {'id': 25, 'tract':'inferior occipito-frontal fascicle', 'side':'left'},
            'IFO_right': {'id': 26, 'tract':'inferior occipito-frontal fascicle', 'side':'right'},
            'ILF_left': {'id': 27, 'tract':'inferior longitudinal fascicle', 'side':'left'},
            'ILF_right': {'id': 28, 'tract':'inferior longitudinal fascicle', 'side':'right'},
            'MCP': {'id': 29, 'tract':'middle cerebellar peduncle', 'side':'NA'},
            'OR_left': {'id': 30, 'tract':'optic radiation', 'side':'left'},
            'OR_right': {'id': 31, 'tract':'optic radiation', 'side':'right'},
            'POPT_left': {'id': 32, 'tract':'parieto‐occipital pontine', 'side':'left'},
            'POPT_right': {'id': 33, 'tract':'parieto‐occipital pontine', 'side':'right'},
            'SCP_left': {'id': 34, 'tract':'superior cerebellar peduncle', 'side':'left'},
            'SCP_right': {'id': 35, 'tract':'superior cerebellar peduncle', 'side':'right'},
            'SLF_I_left': {'id': 36, 'tract':'superior longitudinal fascicle I', 'side':'left'},
            'SLF_I_right': {'id': 37, 'tract':'superior longitudinal fascicle I', 'side':'right'},
            'SLF_II_left': {'id': 38, 'tract':'superior longitudinal fascicle II', 'side':'left'},
            'SLF_II_right': {'id': 39, 'tract':'superior longitudinal fascicle II', 'side':'right'},
            'SLF_III_left': {'id': 40, 'tract':'superior longitudinal fascicle III', 'side':'left'},
            'SLF_III_right': {'id': 41, 'tract':'superior longitudinal fascicle III', 'side':'right'},
            'STR_left': {'id': 42, 'tract':'superior thalamic radiation', 'side':'left'},
            'STR_right': {'id': 43, 'tract':'superior thalamic radiation', 'side':'right'},
            'UF_left': {'id': 44, 'tract':'uncinate fascicle', 'side':'left'},
            'UF_right': {'id': 45, 'tract':'uncinate fascicle', 'side':'right'},
            'T_PREF_left': {'id': 46, 'tract':'thalamo-prefrontal', 'side':'left'},
            'T_PREF_right': {'id': 47, 'tract':'thalamo-prefrontal', 'side':'right'},
            'T_PREM_left': {'id': 48, 'tract':'thalamo-premotor', 'side':'left'},
            'T_PREM_right': {'id': 49, 'tract':'thalamo-premotor', 'side':'right'},
            'T_PREC_left': {'id': 50, 'tract':'thalamo-precentral', 'side':'left'},
            'T_PREC_right': {'id': 51, 'tract':'thalamo-precentral', 'side':'right'},
            'T_POSTC_left': {'id': 52, 'tract':'thalamo-postcentral', 'side':'left'},
            'T_POSTC_right': {'id': 53, 'tract':'thalamo-postcentral', 'side':'right'},
            'T_PAR_left': {'id': 54, 'tract':'thalamo-parietal', 'side':'left'},
            'T_PAR_right': {'id': 55, 'tract':'thalamo-parietal', 'side':'right'},
            'T_OCC_left': {'id': 56, 'tract':'thalamo-occipital', 'side':'left'},
            'T_OCC_right': {'id': 57, 'tract':'thalamo-occipital', 'side':'right'},
            'ST_FO_left': {'id': 58, 'tract':'striato-fronto-orbital', 'side':'left'},
            'ST_FO_right': {'id': 59, 'tract':'striato-fronto-orbital', 'side':'right'},
            'ST_PREF_left': {'id': 60, 'tract':'striato-prefrontal', 'side':'left'},
            'ST_PREF_right': {'id': 61, 'tract':'striato-prefrontal', 'side':'right'},
            'ST_PREM_left': {'id': 62, 'tract':'striato-premotor', 'side':'left'},
            'ST_PREM_right': {'id': 63, 'tract':'striato-premotor', 'side':'right'},
            'ST_PREC_left': {'id': 64, 'tract':'striato-precentral', 'side':'left'},
            'ST_PREC_right': {'id': 65, 'tract':'striato-precentral', 'side':'right'},
            'ST_POSTC_left': {'id': 66, 'tract':'striato-postcentral', 'side':'left'},
            'ST_POSTC_right': {'id': 67, 'tract':'striato-postcentral', 'side':'right'},
            'ST_PAR_left': {'id': 68, 'tract':'striato-parietal', 'side':'left'},
            'ST_PAR_right': {'id': 69, 'tract':'striato-parietal', 'side':'right'},
            'ST_OCC_left': {'id': 70, 'tract':'striato-occipital', 'side':'left'},
            'ST_OCC_right': {'id': 71, 'tract':'striato-occipital', 'side':'right'}
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
            return [path for path in self.hcp_path.glob("*") if path.is_dir()]
        elif self.scope == "trainset":
            # Devuelve la ruta de todas las subcarpetas de la carpeta trainset
            return [path for path in self.hcp_path.glob("*") if path.is_dir() and path.name in self.trainset]
        elif self.scope == "validset":
            # Devuelve la ruta de todas las subcarpetas de la carpeta validationset
            return [path for path in self.hcp_path.glob("*") if path.is_dir() and path.name in self.validset]
        elif self.scope == "testset":
            # Devuelve la ruta de todas las subcarpetas de la carpeta testset
            return [path for path in self.hcp_path.glob("*") if path.is_dir() and path.name in self.testset]
        
    def get_split_from_subject(self, subject:pathlib.Path) -> str:
        """
        Function to get the split of the subject.

        Args:
            subject (str): Name of the subject.

        Returns:
            str: Split of the subject.
        """
        if subject.name in self.trainset:
            return "trainset"
        elif subject.name in self.validset:
            return "validset"
        elif subject.name in self.testset:
            return "testset"
        else:
            return "unknown"
            
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
        # if tract not in self.TRACT_LIST.keys() and tract != "all":
        #     raise ValueError("El tracto no existe.")
        
        if tract == "all":
            # Devuelve la ruta de todos los tractos de un sujeto
            return [path for path in subject.joinpath("tracts").glob(f"*.{extension}")]
        else:
            # Devuelve la ruta de un tracto de un sujeto
            return [path for path in (subject.joinpath("tracts").glob(f"*{tract}*.{extension}"))]    

    def get_data_from_subject(self, subject:pathlib.Path, extension:str = "trk") -> dict:
        split = self.get_split_from_subject(subject)
        tracts = self.get_tract_paths_from_suj(subject, extension = extension) # Devuelve la lista de tractos de un sujeto
        return {"subject": subject.name, "subject_split":split, "T1w": None, "tracts": tracts}
    
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
    trinf_handler = Tractoinferno_handler(path = "/app/dataset/Tractoinferno/derivatives", scope = "testset")
    hcp_handler = HCP_handler(path = "/app/dataset/HCP_105", scope = "testset")
    
    trinf_paths = trinf_handler.get_data()
    hcp_paths = hcp_handler.get_data()

    print(trinf_paths, len(trinf_paths))
    print(hcp_paths, len(hcp_paths))

   