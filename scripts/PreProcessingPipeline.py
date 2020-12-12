from scripts.PreProcessingToolkit import PreProcessingToolkit
import os
import glob
import logging

"""
This is the preprocessing pipeline. It processes the meningioma images so they are ready for
training. The preprocessing steps are as follows:

1) Load nifti
2) Resample nifti to 256x256xZ (height, width, number of slices). Z = 256 if nifti has < 256 slices, or Z if Z slices
3) Get slices that contain the tumor
4) Decompose nifti into individual slices
5) Save each slice as it's own nifti
5) Create labels for each slice

We are using this approach to get more data to train on the CNN as 85 subjects is a relatively low number to train on.
This is essentially augmenting the data.
"""
class PreProcessingPipeline:
    def __init__(self):
        self.input_dir = './Meningioma/MRI/original'
        self.output_dir = './Meningioma/MRI/slices'
        self.labels_dir = './Meningioma/labels'
        self.sequences = ['adc.nii.gz', 'flair.nii.gz', 't1.nii.gz', 't1c.nii.gz', 't2.nii.gz']
        self.lesion_mask = 'lesion.nii.gz'

    def Run(self):
        # Create a new instance of the preprocessing toolkit which contains all the functions needed to
        # preprocess the images and create the labels
        preprocessing = PreProcessingToolkit()

        # Create empty list to store labels
        labels = []

        # Let's use os.walk to search through each subject directory. For each subject directory,
        # we will run preprocessing on each image within the directory.
        for root, dirs, files in os.walk(self.input_dir):
            # Iterate over each subject directory in the input directory
            for dir in dirs:
                # Get all image path's within a subject directory that are of type .nii.gz (nifti file type)
                img_paths = glob.glob(os.path.join(root, dir) + '/' + '*.nii.gz')

                # Trim image path and store only the filename itself without the parent directory path
                imgs = [os.path.basename(image_path) for image_path in img_paths]

                # Check if the subject has a lesion mask. If the subject doesn't have a lesion mask we won't
                # know which slices in the image contain the meningioma. Pass the subject if they have no lesion mask.
                if not self.lesion_mask in imgs:
                    continue

                # Check if the subject has all sequences. If they don't, it can't be used during training
                # because the model will expect a specific number of input channels. One workaround is to create
                # an empty image if the sequence doesn't exist or to copy an image from a different sequence.
                # Would this affect the training though?
                # For now we will pass the subject if they don't have all the required sequences
                if not all(sequence in imgs for sequence in self.sequences):
                    continue

                # In our case, the subject name is the name of the directory
                subject = dir

                logging.debug("Subject: " + subject)
                logging.debug("Number of Images: " + str(len(imgs)))
                logging.debug("Images: " + str(imgs))

                # Find all masks in the subject directory (there should only be 1 really). If there are multiple,
                # use the first one.
                mask_path = [i for i in img_paths if self.lesion_mask in i][0]
                logging.debug("Mask: " + str(os.path.basename(mask_path)))

                # Load the nifti file. The return mask is of type nibabel
                mask = preprocessing.load_nifti(mask_path)

                # Resample nifti to 256x256xZ (height, width, number of slices).
                # Z = 256 if nifti has < 256 slices, or Z if Z slices.
                mask = preprocessing.resample(mask)

                # Find all image paths (that are not a mask) in the list of image paths
                img_paths = [i for i in img_paths if not self.lesion_mask in i]

                # Iterate through each image path and apply preprocessing on the image
                for img_path in img_paths:
                    # Image type contained within the filename
                    img_type = os.path.basename(img_path)
                    logging.debug("Image Type: " + str(img_type))
                    logging.debug("")

                    # Load nifti file
                    img = preprocessing.load_nifti(img_path)

                    # Apply n4 bias field correction (I never did this because I haven't figured out how to convert
                    # the return image file into a pixel array)
                    # img = preprocessing.apply_n4biasfieldcorrection(img)

                    # Resample nifti to 256x256xZ (height, width, number of slices).
                    # Z = 256 if nifti has < 256 slices, or Z if Z slices.
                    # Note, The mask has the same original dimensions as the other images. Otherwise the resampled
                    # mask would not be in sync with the other resampled images. This would mess everything up.
                    # (fyi, I did check that the shapes for all images within a subject were the same but code
                    # is not in any of these scripts).
                    img = preprocessing.resample(img)

                    # Get slice numbers containing tumor
                    lesion_slices = preprocessing.get_lesion_slices(mask=mask)

                    # Decompose nifti into individual slices
                    img_slices = preprocessing.decompose_img(img=img)

                    # Save each slice as it's own nifti
                    img_indices, img_ids = preprocessing.save_slices(self.output_dir, subject, img_type, img_slices)

                    # Create labels for all slices
                    labels.append(preprocessing.get_labels(subject, img_indices, img_ids, lesion_slices))

        # Create dataframe from labels
        labels_df = preprocessing.labels_to_df(labels)

        # Save labels to csv
        labels_df.to_csv(os.path.join(self.labels_dir, 'data.csv'), index=False)
