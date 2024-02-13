import pathlib2 as pathlib
from dataset_handlers import Dataset_handler, Tractoinferno_handler
import numpy as np
import nibabel as nib
from dipy.data.fetcher import fetch_mni_template, read_mni_template
from dipy.align.imaffine import AffineMap, MutualInformationMetric, AffineRegistration
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D
from dipy.align.reslice import reslice
from dipy.io.streamline import load_trk
from dipy.tracking.streamline import transform_streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
import multiprocessing as mp


class MNI_preprocessor:
    """
    Clase para preprocesar los archivos 
    """
    def __init__(self):
        self.template = self.load_mni_template()# Cargar la plantilla MNI


    def load_mni_template(self, 
                          save:bool = False, 
                          save_path = './mni_t1w_template.nii.gz') -> nib.Nifti1Image:
        """
        Loads the MNI template and returns it as a NIfTI image.

        Parameters:
        - save (bool): Whether to save the MNI template as a NIfTI file. Default is False.

        Returns:
        - image (nib.Nifti1Image): The loaded MNI template as a NIfTI image.
        """
        fetch_mni_template() # Fetch the MNI 2009a T1 and T2, and 2009c T1 and T1 mask files
        T1w, mask = read_mni_template(version="c", contrast=["T1", "mask"])
        image =  nib.Nifti1Image(T1w.get_fdata() * mask.get_fdata(), T1w.affine, T1w.header)
        
        if save:# Save the MNI template as a NIfTI file
            nib.save(image, save_path)

        return image

    def resample_to_mni(self, data:nib.Nifti1Image) -> nib.Nifti1Image:
        """
        Resample the data to the MNI template.

        Args:
            data (nib.Nifti1Image): The data to resample.
            template (nib.Nifti1Image): The template to resample to. Default is the MNI template.

        Returns:
            nib.Nifti1Image: The resampled data.
        """
        # Check if data is a NIfTI image and if not, raise an error
        if not isinstance(data, nib.Nifti1Image):
            raise ValueError("Data must be a NIfTI image.")

        # Check if is necessary to resample
        if data.header.get_zooms()[:3] == self.template.header.get_zooms()[:3]:
            return data
        else:
            # Resample data to 1mm x 1mm x 1mm
            data_resampled, affine = reslice(data.get_fdata(), 
                                             data.affine, 
                                             data.header.get_zooms()[:3], 
                                             self.template.header.get_zooms()[:3])
            data_resampled = nib.Nifti1Image(data_resampled, affine)

            return data_resampled

    
    
    def mni_register(self, subject:pathlib.Path) -> tuple[nib.Nifti1Image, np.ndarray]:
        """
        Registra una imagen de un sujeto a la plantilla MNI usando una transformación T, calculada 
        a traves de un proceso de registro traslacion, registro rigido y registro afin.

        Args:
            subject (pathlib.Path): Path de la imagen del sujeto a registrar.

        Returns:
            registered_data (numpy.ndarray): Imagen registrada.
            transformation_matrix (numpy.ndarray): Matriz de transformación.
        """
        # Load mni template and subject image
        template_data = self.template.get_fdata() # Image (voxel grid)
        template_affine = self.template.affine # Affine matrix

        moving_img = nib.load(subject)
        moving_data = moving_img.get_fdata()
        moving_affine = moving_img.affine

        # The mismatch metric
        metric = MutualInformationMetric(nbins = 32, 
                                        sampling_proportion = None)
        
        # The optimization strategy
        affreg = AffineRegistration(metric = metric,
                                    level_iters = [10, 10, 5],
                                    sigmas = [3.0, 1.0, 0.0],
                                    factors = [4, 2, 1])
        
        params0 = None
        
        # print("Starting translation registration")
        translation_transform = TranslationTransform3D()
        translation = affreg.optimize(static = template_data, 
                                    moving = moving_data, 
                                    transform = translation_transform, 
                                    params0 = params0,
                                    static_grid2world = template_affine, 
                                    moving_grid2world = moving_affine)
        
        # print("Starting rigid registration")
        rigid_transform = RigidTransform3D()
        rigid = affreg.optimize(static = template_data, 
                                    moving = moving_data, 
                                    transform = rigid_transform, 
                                    params0 = params0,
                                    static_grid2world = template_affine, 
                                    moving_grid2world = moving_affine,
                                    starting_affine=translation.affine)
        
        # print("Starting affine registration")
        affine_transform = AffineTransform3D()
        affreg.level_iters = [1000, 1000, 100]
        registration = affreg.optimize(static = template_data,
                                moving = moving_data,
                                transform = affine_transform,
                                params0 = params0,
                                static_grid2world = template_affine,
                                moving_grid2world = moving_affine,
                                starting_affine = rigid.affine)
        
        # Transform the input image
        transformation_matrix = registration.affine
        registered_data = registration.transform(image = moving_data,
                                                image_grid2world = moving_affine,
                                                sampling_grid_shape = template_data.shape)
        
        registered_data = nib.Nifti1Image(registered_data, template_affine)

        return registered_data, transformation_matrix
    
    def register_streamlines(self, 
                             tract_file:pathlib.Path, 
                             transform_matrix:np.ndarray) -> StatefulTractogram:
        """
        Registra un tracto en el espacio MNI usando una matriz de transformación calculada.
        """
        # Load trk file
        tract = load_trk(str(tract_file), 'same')
        tract_streamlines = tract.streamlines

        # Transform streamlines to target space
        tract_streamlines = transform_streamlines(streamlines = tract_streamlines, 
                                                  mat = np.linalg.inv(transform_matrix))
        
        sft = StatefulTractogram(streamlines = tract_streamlines, 
                                reference = self.template, 
                                space = Space.RASMM)
            
        return sft
    

    # Funcion para preprocesar los archivos de un sujeto estructurado bids y guardarlos en la carpeta tractoinferno_preprocessed_mni
    def preprocess_subject(self,
                           data:dict,
                           directorio_raiz_destino:pathlib.Path = None,
                           verbose:bool = False,
                           bbox_valid_check:bool = False,
                           output_format:str = "trk"):
        """
        Funcion para preprocesar un sujeto y guardarlo en la carpeta tractoinferno_preprocessed_mni.
        """
        
        subject_id = data["subject"]
        subject_split = data["subject_split"]
        anat = data["T1w"]
        tracts = data["tracts"]
        # Generar todas las rutas de destino de los archivos
        # El directorio de destino tiene tres subcarpetas: trainset, validset y testset, a su vez, cada carpeta contiene una subcarpeta sujeto y dentro las subcarpetas anat y tractography
        
        # Rutas de destino segun estandar BIDS
        # Guarda la anatomia en la carpeta anat/sujeto-001__mni.nii.gz y los tractos en la carpeta tractography/tractoAF_L__mni.trk
        anat_destino = directorio_raiz_destino.joinpath(subject_split, subject_id, "anat", f"{anat.stem}__mni.nii.gz")
        tract_destino = [directorio_raiz_destino.joinpath(subject_split, subject_id, "tractography", f"{tract.stem}__mni.{output_format}") for tract in tracts]
        
        # Registrar imagen anat y resamplearla a la plantilla MNI
        reg_subject, transformation_mat = self.mni_register(subject = anat)
        reg_subject = self.resample_to_mni(reg_subject)

        # Crear directorio de destino
        directorio_sujeto_destino = directorio_raiz_destino.joinpath(subject_split, subject_id)
        directorio_anat_destino = directorio_sujeto_destino.joinpath("anat")
        directorio_tractography_destino = directorio_sujeto_destino.joinpath("tractography")

        directorio_sujeto_destino.mkdir(parents=True, exist_ok=True)
        directorio_anat_destino.mkdir(parents=True, exist_ok=True)
        directorio_tractography_destino.mkdir(parents=True, exist_ok=True)

        
        nib.save(reg_subject, anat_destino)# Guardar imagen anat
        if verbose:
            print(f"Imagen t1w guardada en {anat_destino}")
        
        # Guardar tractos
        for idx, tract in enumerate(tracts):
            sft = self.register_streamlines(tract, transformation_mat)
            save_tractogram(sft, 
                            str(tract_destino[idx]), 
                            bbox_valid_check = bbox_valid_check)
            if verbose:
                print(f"Tracto {tract.stem} guardado en {tract_destino[idx]} correctamente.")
        return directorio_sujeto_destino
    

    def preprocess_set_of_subjects(self,
                                   file_list:list[dict], 
                                   dir_destino:pathlib.Path, 
                                   num_workers = 1):
        """
        Funcion para registrar en paralelo una lista de sujetos
        """
        
        # paths = [dataset_handler.get_data_from_subject(subject) for subject in dataset_handler.subjects]
        print(f"Se preprocesarán {len(file_list)} sujetos.")
        
        # Si num_workers es -1, se utiliza el numero de procesadores disponibles
        if num_workers == 1:
            [self.preprocess_subject(subject, dir_destino, True) for subject in file_list]
        else:
            if num_workers == -1 or num_workers > mp.cpu_count():
                num_workers = mp.cpu_count() - 2 # Se deja un procesador libre para el sistema
            
            elif num_workers < mp.cpu_count() and num_workers > 0:
                num_workers = num_workers
            
            else:# Error en el numero de workers
                raise ValueError("El numero de workers debe ser mayor que 0.")   

            print(f"Se utilizaran {num_workers} procesadores.")    
            
            with mp.Pool(processes=num_workers) as pool:
                pool.starmap(self.preprocess_subject, [(subject, dir_destino, True) for subject in file_list])
                pool.close()

            print("Procesamiento finalizado.")


        return None
        


if __name__ == "__main__":
    # Cargar el dataset
    # validset_handler = Tractoinferno_handler(tractoinferno_path = r"C:\Users\pablo\Documents\Datasets\ds003900-download\derivatives",
    #                                         scope="validset")
    # trainset_handler = Tractoinferno_handler(tractoinferno_path = r"C:\Users\pablo\Documents\Datasets\ds003900-download\derivatives",
    #                                         scope="trainset")
    
    # Preprocesar el conjunto de test del dataset 
    testset_handler = Tractoinferno_handler(tractoinferno_path = r"C:\Users\pablo\Documents\Datasets\ds003900-download\derivatives",
                                            scope="testset")
    testset_data = testset_handler.get_data()# list[dict{"subject":str, "subject_split":str, "T1w":Path, "tracts":[Path]}]

    # Preprocesar el dataset
    MNI_preprocessor().preprocess_set_of_subjects(file_list = testset_data, 
                                                  dir_destino = pathlib.Path(r"C:\Users\pablo\GitHub\tfm_prg\tractoinferno_preprocessed_mni"), 
                                                  num_workers = -1)
    

