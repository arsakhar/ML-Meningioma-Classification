import SimpleITK as sitk
import nibabel as nib
from nibabel import processing
import numpy as np
import os
import itertools
import pandas as pd
import logging

class PreProcessingToolkit:
    def __init__(self):
        pass

    """
    Use nibabel to load image  
    ================== ===========================================================================
    **Arguments:**
    img_path           the image path
    
    **Returns:**
    img                image (nibabel type)
    ================== ===========================================================================
    """
    def load_nifti(self, img_path):
        img = nib.load(img_path)

        return img

    """
    Resample image to 256x256xZ (height, width, number of slices).
    Z = 256 if image has < 256 slices, or Z if Z slices.
    ================== ===========================================================================
    **Arguments:**
    img                image

    **Returns:**
    img                image (resampled)
    ================== ===========================================================================
    """
    def resample(self, img):
        if img.shape[2] > 256:
            img_resampled = processing.conform(img, out_shape=(256, 256, img.shape[2]))

        else:
            img_resampled = processing.conform(img, out_shape=(256, 256, 256))

        logging.debug("Resampling: ")
        logging.debug("""
        Original image shape : {0}
        Resampled image shape: {1}
        """.format(img.shape, img_resampled.shape))

        return img_resampled

    """
    Filter the image by slices containing tumors.
    NOTE: I'm leaving this function here but it isn't used.
    ================== ===========================================================================
    **Arguments:**
    img                image
    mask               mask

    **Returns:**
    img                image (filtered)
    ================== ===========================================================================
    """
    def filter_by_lesion_slices(self, img, mask):
        logging.debug("Filtering Image by Lesion Slices: ")
        # print("Filtering Image by Lesion Slices: ")

        if img.shape == mask.shape:
            # gets the slices numbers that have tumors in the mask
            mask_indices = self.get_lesion_slices(mask)

            # get ndarray from nibabel nifti
            img_data = img.get_fdata()

            # take only the slices denoted by mask indices. 2 denotes the 3rd dimension (slice dimension)
            img_data = np.take(img_data, mask_indices, 2)

            # convert ndarray back to nifti image
            img_filtered = nib.Nifti1Image(img_data, img.affine)

        else:
            logging.debug("mask and image shape do not match")
            logging.debug("""
            image shape : {0}
            mask shape: {1}
            """.format(img.shape, mask.shape))

        return img_filtered

    """
    Get all mask slices that contain the tumor.
    ================== ===========================================================================
    **Arguments:**
    mask               mask

    **Returns:**
    lesion_slices      list of slice numbers containing tumor
    ================== ===========================================================================
    """
    def get_lesion_slices(self, mask):
        logging.debug("Getting Lession Slices: ")

        lesion_slices = []

        # get ndarray from nibabel nifti
        mask_data = mask.get_fdata()

        # iterate over number of slices in mask
        for slice in range(mask_data.shape[2]):

            # get max pixel intensity of image slice
            max_pixel_intensity = np.amax(mask_data[:, :, slice])

            # if the image slice isn't empty (black), add slice number to list
            if max_pixel_intensity > 0:
                lesion_slices.append(slice)

        logging.debug(str(lesion_slices))
        logging.debug("")

        return lesion_slices

    """
    Decompose the nifti into individual slices
    ================== ===========================================================================
    **Arguments:**
    img                image

    **Returns:**
    imgs               list of nibabel images
    ================== ===========================================================================
    """
    def decompose_img(self, img):
        # convert nibabel nifti to ndarray
        img_data = img.get_fdata()

        imgs = []
        for slice in range(img_data.shape[2]):
            img_slice = img_data[:, :, slice]

            img_slice = nib.Nifti1Image(img_slice, img.affine)

            imgs.append(img_slice)

        return imgs

    """
    Apply bias field correction to image
    ================== ===========================================================================
    **Arguments:**
    img_path           the image path

    **Returns:**
    img                image of type SimpleITK
    ================== ===========================================================================
    """
    def apply_n4biasfieldcorrection(self, img_path):
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        maskImage = sitk.OtsuThreshold(img, 0, 1, 200)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        img = corrector.Execute(img, maskImage)

        return img

    """
    Save image slices as their own nifti file
    ================== ===========================================================================
    **Arguments:**
    dir                root directory to save image slices
    subject            subject id
    img_type           image slices type
    img_slices         a list of images (slice)

    **Returns:**
    img_indices        the indices of the saved image slices
    img_ids            the ids of the saved image slices
    ================== ===========================================================================
    """
    def save_slices(self, dir, subject, img_type, img_slices):
        # define subject directory
        subject_dir = os.path.join(dir, subject)

        # create subject directory if it doesn't exist (it shouldn't if this is the first time running preprocessing)
        if not os.path.exists(subject_dir):
            os.mkdir(subject_dir)

        img_ids = []
        img_indices = []
        # iterate over each image slice in the list
        for img_index, img_slice in enumerate(img_slices):
            # convert nibabel nifti slice to ndarray
            slice_data = img_slice.get_fdata()

            # get max pixel intensity of image slice
            max_pixel_intensity = np.amax(slice_data)

            # if the image isn't empty (black), save it and store image index and id
            if max_pixel_intensity > 0:
                slice_id = subject + '-' + str(img_index) + '-' + img_type
                nib.save(img_slice, os.path.join(dir, subject, slice_id))
                img_indices.append(img_index)
                img_ids.append(slice_id)

        return img_indices, img_ids

    """
    Get labels
    Returns a list of list of labels.
    Format: image id, label
    ================== ===========================================================================
    **Arguments:**
    subject            subject id
    img_indices        the indices of the saved image slices
    img_ids            the ids of the saved image slices
    lesion_indices     the indices of the tumor slices

    **Returns:**
    labels             a list of list of labels
    ================== ===========================================================================
    """
    def get_labels(self, subject, img_indices, img_ids, lesion_indices):
        labels = []

        for img_index, img_id in zip(img_indices, img_ids):
            # if the image slice is a tumor slice, set label as subject[0] (first character in subject ID)
            # In our case, all subjects are named as "X_ID" where X is the label so first character is label
            if img_index in lesion_indices:
                labels.append([subject + '/' + img_id, subject[0]])

            # if it isn't a tumor slice, assign a -1 label to indicate no tumor
            else:
                labels.append([subject + '/' + img_id, -1])

        return labels

    """
    Convert labels list to dataframe
    ================== ===========================================================================
    **Arguments:**
    labels             a list of list of labels

    **Returns:**
    labels_df          a labels dataframe
    ================== ===========================================================================
    """
    def labels_to_df(self, labels):
        labels = list(itertools.chain(*labels))
        labels_df = pd.DataFrame(labels, columns=['id', 'classification'])

        return labels_df

    """
    Convert labels dataframe from a single column of -1, 0, 1, 2, or 3 into a seperate column
    for each label type.
    So dataframe header would be: id, -1, 0, 1, 2, 3
    An example row would be:  0_S021/0_S021-55-adc.nii.gz, 0, 1, 1, 1, 0 if 2 was the correct grade
    for that image. This is because we are using thermometer encoding where the image hardness is
    all values up to and including it's actual hardness
    ================== ===========================================================================
    **Arguments:**
    labels_df          a labels dataframe

    **Returns:**
    labels_df          a labels dataframe
    ================== ===========================================================================
    """
    def ordinal_encoding(self, labels_df):
        # convert classification column to int
        labels_df['classification'] = labels_df['classification'].astype(int)

        # create a list of unique classes sorted by rank (-1 to 3)
        classes = sorted(labels_df['classification'].unique().tolist())

        labels = []
        # iterate over each row in the dataframe
        for index, row in labels_df.iterrows():
            # create a list of 0's that is the number of classes - 1 (i.e. [0, 0, 0, 0])
            # why did I do -1 again?
            encodings = [0] * (len(classes) - 1)

            # row['classification'] gets 'class' for that image. index gets position of 'class' in list
            # end position represents the actual label for the image
            end_pos = classes.index(row['classification'])

            # if end pos > 0 means if the class was 0-3 and not -1, we want to fill the encodings list
            # with all 1's up to the end-pos.
            if end_pos > 0:
                encodings[0:end_pos] = [1] * (end_pos - 0)

            encodings.insert(0, row['id'])
            labels.append(encodings)

        classes[0] = 'id'
        labels_df = pd.DataFrame(labels, columns=classes)

        return labels_df

    """
    Convert labels dataframe from a single column of -1, 0, 1, 2, or 3 into a seperate column
    for each label type.
    So dataframe header would be: id, -1, 0, 1, 2, 3
    An example row would be:  0_S021/0_S021-55-adc.nii.gz, 0, 0, 0, 1, 0 if 2 was the correct grade
    for that image
    ================== ===========================================================================
    **Arguments:**
    labels_df          a labels dataframe

    **Returns:**
    labels_df          a labels dataframe
    ================== ===========================================================================
    """
    def onehot_encoding(self, labels_df):
        # convert classification column to int
        labels_df['classification'] = labels_df['classification'].astype(int)

        # create a list of unique classes sorted by rank (-1 to 3)
        classes = sorted(labels_df['classification'].unique().tolist())

        labels = []
        # iterate over each row in the dataframe
        for index, row in labels_df.iterrows():
            # create a list of 0's that is the number of classes (i.e. [0, 0, 0, 0, 0])
            encodings = [0] * (len(classes))

            # index is position in classes list containing 'class' associated with row
            index = classes.index(row['classification'])

            # set encoding list to 1 at this position
            encodings[index] = 1

            encodings.insert(0, row['id'])
            labels.append(encodings)

        classes.insert(0, 'id')
        labels_df = pd.DataFrame(labels, columns=classes)

        return labels_df