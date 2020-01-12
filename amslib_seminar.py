import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import keras
from keras import backend as K

def resample_image(input_image, spacing_mm=(1, 1, 1), spacing_image=None, inter_type=sitk.sitkLinear):
    """
    Resample image to desired pixel spacing.

    Should specify destination spacing immediate value in parameter spacing_mm or as SimpleITK.Image in spacing_image.
    You must specify either spacing_mm or spacing_image, not both at the same time.

    :param input_image: Image to resample.
    :param spacing_mm: Spacing for resampling in mm given as tuple or list of two/three (2D/3D) float values.
    :param spacing_image: Spacing for resampling taken from the given SimpleITK.Image.
    :param inter_type: Interpolation type using one of the following options:
                            SimpleITK.sitkNearestNeighbor,
                            SimpleITK.sitkLinear,
                            SimpleITK.sitkBSpline,
                            SimpleITK.sitkGaussian,
                            SimpleITK.sitkLabelGaussian,
                            SimpleITK.sitkHammingWindowedSinc,
                            SimpleITK.sitkBlackmanWindowedSinc,
                            SimpleITK.sitkCosineWindowedSinc,
                            SimpleITK.sitkWelchWindowedSinc,
                            SimpleITK.sitkLanczosWindowedSinc
    :type input_image: SimpleITK.Image
    :type spacing_mm: Tuple[float]
    :type spacing_image: SimpleITK.Image
    :type inter_type: int
    :rtype: SimpleITK.Image
    :return: Resampled image as SimpleITK.Image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(inter_type)

    if (spacing_mm is None and spacing_image is None) or \
       (spacing_mm is not None and spacing_image is not None):
        raise ValueError('You must specify either spacing_mm or spacing_image, not both at the same time.')

    if spacing_image is not None:
        spacing_mm = spacing_image.GetSpacing()

    input_spacing = input_image.GetSpacing()
    # set desired spacing
    resampler.SetOutputSpacing(spacing_mm)
    # compute and set output size
    output_size = np.array(input_image.GetSize()) * np.array(input_spacing) \
                  / np.array(spacing_mm)
    output_size = list((output_size + 0.5).astype('uint32'))
    output_size = [int(size) for size in output_size]
    resampler.SetSize(output_size)

    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())

    resampled_image = resampler.Execute(input_image)

    return resampled_image

def load_mri_brain_data_masks():
    """
    Load masks from 'data' folder.
    """
    # hidden function parameters
    DATA_PATH = 'data'
    
    # load and extract all images and masks into a list of dicts
    mri_data = []

    patient_paths = os.listdir(DATA_PATH)
    for patient_no in tqdm(range(len(patient_paths))):

        # Generacija imen map.
        patient_path_masks=os.path.join(DATA_PATH,patient_paths[patient_no],'masks')
    
        # Generacija imen datotek v mapah.
        mask1_01_string='_'.join([patient_paths[patient_no],'01','mask1.nii.gz'])
        mask1_02_string='_'.join([patient_paths[patient_no],'02','mask1.nii.gz'])
    
        # read all masks
        mask1_01 = sitk.ReadImage(os.path.join(patient_path_masks,mask1_01_string))
        mask1_02 = sitk.ReadImage(os.path.join(patient_path_masks,mask1_02_string))
        
        # add to dict
        mri_data.append({'mask1_01':mask1_01, 'mask1_02': mask1_02})
        
    # reshape all masks into 4d arrays
    mask1_01_array = np.stack([sitk.GetArrayFromImage(data['mask1_01']) for data in mri_data])
    mask1_02_array = np.stack([sitk.GetArrayFromImage(data['mask1_02']) for data in mri_data])     

    # return the brainmasks
    return mask1_01_array, mask1_02_array

def load_mri_brain_data_classifications(dV_threshold=0,dtype='float64'):
    
    """
    Load masks from 'data' folder and extracts classification information.
    """
    
    # hidden function parameters
    DATA_PATH = 'data'
    
    # load and extract all images and masks into a list of dicts
    dV = []

    patient_paths = os.listdir(DATA_PATH)
    for patient_no in tqdm(range(len(patient_paths))):

        # Generacija imen map.
        patient_path_masks=os.path.join(DATA_PATH,patient_paths[patient_no],'masks')
    
        # Generacija imen datotek v mapah.
        mask1_01_string='_'.join([patient_paths[patient_no],'01','mask1.nii.gz'])
        mask1_02_string='_'.join([patient_paths[patient_no],'02','mask1.nii.gz'])
    
        # read all masks
        mask1_01 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_path_masks,mask1_01_string)))
        mask1_02 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_path_masks,mask1_02_string)))
        
        # compute volumes
        V1=np.sum(mask1_01==1)/1000 # including conversion from mm**3 to ml
        V2=np.sum(mask1_02==1)/1000 # including conversion from mm**3 to ml
        dV.append(V2-V1)
     
    dV
    # Pretvorba vektorja oznak razreda v binarno matriko oznak tipa 1-k
    Y=keras.utils.to_categorical((np.asarray(dV)>dV_threshold).astype('int')).astype(dtype)
    
    return Y
    
def load_mri_brain_data(output_size=(64, 64, 64), no_slices=192, modalities=('t1_01', 'flair_01', 'mask1_01', 't1_02', 'flair_02', 'mask1_02'), dtype='float64'):
    """
    Load data from 'data' folder in the format suitable for training neural networks.

    This functions will load all images, perform cropping to fixed size and resample the obtained image such
    that the output size will match the one specified by parameter 'output_size'. The data output will contain the
    modalities as specified by parameter 'modalities'.

    :param output_size: Define output image size.
    :param modalities: List of possible modalities specified by strings 't1_01', 'flair_01', 'mask1_01', 't1_02', 'flair_02' and 'mask1_02'.
    :type output_size: tuple[int]
    :type modalities: tuple[str]
    :rtype: numpy.ndarray
    :return: Image data.
    """
    # hidden function parameters
    DATA_PATH = 'data'

    # define image extraction function based on cropping and resampling
    # OPOMBA (obrezovanje): obrezovanje po aksialni osi opravimo glede na število rezin no_slices, ki jih želimo izločiti. V primeru, da obrežemo stran sodo število rezino, bo funkcija z obeh koncev
    # obrezala polovico navedenih rezin. V primeru, da obrežemo stran liho število rezin, bo funkcija na koncu, kjer je večji indeks, obrezala eno rezino več.
    # OPOMBA (vzorčenje): če želimo še vedno imeti enak korak vzorčenja po vseh treh dimenzijah, moramo kot število aksialnih rezin no_slices podati N-krat več rezin, kot je število želenih končnih
    # rezin, pri čemer je N faktor zmanjšanja velikosti slike po ostalih dveh dimenzijah. Primer: če zmanjšamo velikost slike po X in Y dimenziji za faktor 192/64=3 in če želimo imeti 3 rezine po
    # aksialni (Z) osi, moramo za vrednost no_slices podati vrednost 9. Vrednost no_slices bi bilo sicer možno tudi fiksirati tako, da bi jo izračunali glede na podano končno aksialno dimenzijo
    # slike.
    # OPOMBA: vrstni red dimenzij pri slikovnem formatu (X_image.GetSize())numpy je zamenjan v primerjavi z vrstnim redom pri podatkovnem polju (X_array.shape).
    def extract_image(image, output_size=(128, 128, 128), no_slices=192, interpolation_type=sitk.sitkLinear):
        new_spacing_mm = (192 / output_size[0], 192 / output_size[1], no_slices / output_size[2])
        return resample_image(
            sitk.RegionOfInterest(image, (192, 192, no_slices), (0, 18, np.int((193-no_slices)/2))), 
            spacing_mm = new_spacing_mm, 
            inter_type=interpolation_type)
    
    # load and extract all images and masks into a list of dicts
    mri_data = []

    patient_paths = os.listdir(DATA_PATH)
    for patient_no in tqdm(range(len(patient_paths))):

        # Generation of folder path names.
        patient_path_preprocessed=os.path.join(DATA_PATH,patient_paths[patient_no],'preprocessed')
        patient_path_masks=os.path.join(DATA_PATH,patient_paths[patient_no],'masks')
    
        # Generation of file names.
        t1_01_string='_'.join([patient_paths[patient_no],'01','t1w_pp.nii.gz'])
        flair_01_string='_'.join([patient_paths[patient_no],'01','flair_pp.nii.gz'])
        mask1_01_string='_'.join([patient_paths[patient_no],'01','mask1.nii.gz'])
        t1_02_string='_'.join([patient_paths[patient_no],'02','t1w_pp.nii.gz'])
        flair_02_string='_'.join([patient_paths[patient_no],'02','flair_pp.nii.gz'])
        mask1_02_string='_'.join([patient_paths[patient_no],'02','mask1.nii.gz'])
    
        # Loading image files.
        t1_01 = sitk.ReadImage(os.path.join(patient_path_preprocessed,t1_01_string))
        flair_01 = sitk.ReadImage(os.path.join(patient_path_preprocessed,flair_01_string))
        mask1_01 = sitk.ReadImage(os.path.join(patient_path_masks,mask1_01_string))
        t1_02 = sitk.ReadImage(os.path.join(patient_path_preprocessed,t1_02_string))
        flair_02 = sitk.ReadImage(os.path.join(patient_path_preprocessed,flair_02_string))
        mask1_02 = sitk.ReadImage(os.path.join(patient_path_masks,mask1_02_string))
    
        # crop and resample the images
        t1_01 = extract_image(t1_01, output_size, no_slices, sitk.sitkLinear)
        flair_01 = extract_image(flair_01, output_size, no_slices, sitk.sitkLinear)
        mask1_01 = extract_image(mask1_01, output_size, no_slices, sitk.sitkNearestNeighbor)
        t1_02 = extract_image(t1_02, output_size, no_slices, sitk.sitkLinear)
        flair_02 = extract_image(flair_02, output_size, no_slices, sitk.sitkLinear)
        mask1_02 = extract_image(mask1_02, output_size, no_slices, sitk.sitkNearestNeighbor)
        
        # Converting images to numpy arrays of a selected data format.
        t1_01_array_one = sitk.GetArrayFromImage(t1_01).astype(dtype)
        flair_01_array_one = sitk.GetArrayFromImage(flair_01).astype(dtype)
        mask1_01_array_one = sitk.GetArrayFromImage(mask1_01).astype(dtype)
        t1_02_array_one = sitk.GetArrayFromImage(t1_02).astype(dtype)
        flair_02_array_one = sitk.GetArrayFromImage(flair_02).astype(dtype)
        mask1_02_array_one = sitk.GetArrayFromImage(mask1_02).astype(dtype)
        
        # add to dict
        mri_data.append({'t1_01':t1_01_array_one,'flair_01':flair_01_array_one,'mask1_01':mask1_01_array_one,'t1_02':t1_02_array_one,'flair_02':flair_02_array_one,'mask1_02':mask1_02_array_one})
        
    # reshape all modalities and masks into 4d arrays
    t1_01_array = np.stack([data['t1_01'] for data in mri_data])
    flair_01_array = np.stack([data['flair_01'] for data in mri_data])
    mask1_01_array = np.stack([data['mask1_01'] for data in mri_data])
    t1_02_array = np.stack([data['t1_02'] for data in mri_data])
    flair_02_array = np.stack([data['flair_02'] for data in mri_data])
    mask1_02_array = np.stack([data['mask1_02'] for data in mri_data])
    # OPOMBA: dimenzija polja mora imeti tako obliko, da se število pacientov nahaja v prvi dimenziji. V nasprotnem primeru bi moralo dimenzije preoblikovati z numpy funkcijo transpose().
    # Primer: t1_01_array=np.transpose(t1_01_array,(3,0,1,2)). V našem primeru z numpy funkcijo stack že dobimo želeno razporeditev dimenzij.
     
    # reshape the 3d arrays according to the Keras backend
    if K.image_data_format() == 'channels_first':
        # this format is (n_cases, n_channels, image_depth, image_height, image_width)
        t1_01_karray = t1_01_array[:, np.newaxis, :, :, :]
        flair_01_karray = flair_01_array[:, np.newaxis, :, :, :]
        mask1_01_karray = mask1_01_array[:, np.newaxis, :, :, :]
        t1_02_karray = t1_02_array[:, np.newaxis, :, :, :]
        flair_02_karray = flair_02_array[:, np.newaxis, :, :, :]
        mask1_02_karray = mask1_02_array[:, np.newaxis, :, :, :]
        channel_axis = 1
    else:
        # this format is (n_cases, image_height, image_width, n_channels)
        t1_01_karray = t1_01_array[:, :, :, :, np.newaxis]
        flair_01_karray = flair_01_array[:, :, :, :, np.newaxis]
        mask1_01_karray = mask1_01_array[:, :, :, :, np.newaxis]
        t1_02_karray = t1_02_array[:, :, :, :, np.newaxis]
        flair_02_karray = flair_02_array[:, :, :, :, np.newaxis]
        mask1_02_karray = mask1_02_array[:, :, :, :, np.newaxis]
        channel_axis = -1
    
    # Oblikovanje zbirke slik v večdimenzionalno polje.
    mri_data_karray={'t1_01':t1_01_karray, 'flair_01':flair_01_karray, 'mask1_01':mask1_01_karray, 't1_02':t1_02_karray, 'flair_02': flair_02_karray, 'mask1_02': mask1_02_karray}
    mri_data_karray_selected={}
    for key_name in modalities:
        if key_name in mri_data_karray.keys():
            mri_data_karray_selected[key_name]=mri_data_karray[key_name]
        else:
            raise ValueError('The input modalities "{}" are not recognized!'.format(modalities))    
    data=np.concatenate((list(mri_data_karray_selected.values())),axis=channel_axis)      
    
    # read image sizes and channel number
    _, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = data.shape

    # compute min and max values per each channel
    def stat_per_channel(values, stat_fcn):
        return stat_fcn(
            np.reshape(
                values, 
                (values.shape[0]*IMG_DEPTH*IMG_HEIGHT*IMG_WIDTH, IMG_CHANNELS)), 
            axis=0)[:, np.newaxis]

    min_data, max_data = stat_per_channel(data, np.min), stat_per_channel(data, np.max)
    min_data = np.reshape(min_data, (1, 1, 1, 1, IMG_CHANNELS))
    max_data = np.reshape(max_data, (1, 1, 1, 1, IMG_CHANNELS))
    
    # normalize image intensities to interval [0, 1]
    X = (data - min_data) / (max_data - min_data)

    # return the image modalities and brainmasks
    return X, mask1_01_karray, mask1_02_karray