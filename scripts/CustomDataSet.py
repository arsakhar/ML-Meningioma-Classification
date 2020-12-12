from torch.utils.data import Dataset
import os
import nibabel as nib
import torch
import numpy as np
from torchvision import transforms
import torchio as tio
from torchvision.utils import save_image


class CustomDataSet(Dataset):
    def __init__(self, data_df, sequences, target_cols, augmentation=False):
        self.img_dir = './Meningioma/MRI/slices'
        self.data_df = data_df.reset_index(drop=True)
        self.id = 'id'
        self.sequences = sequences
        self.target_cols = target_cols
        self.augmentation = augmentation
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.data_df)

    """
    Returns an image (ndarray) given a sequence and dataframe index
    ================== ===========================================================================
    **Arguments:**
    sequence           image type (sequence)
    index              dataframe index
    
    **Returns:**
    img                image (ndarray)
    ================== ===========================================================================
    """
    def get_image(self, sequence, index):
        imgPath = self.data_df.loc[index, self.id]

        # separate path from sequence
        imgPath = imgPath.split('-')[0:-1]
        imgPath = '-'.join(imgPath)

        # add sequence to path to define new path
        imgPath = imgPath + '-' + sequence

        imgPath = os.path.join(self.img_dir, imgPath)

        try:
            img = nib.load(imgPath)
            img = img.get_fdata()

        except:
            img = np.zeros(shape=(256, 256), dtype="float64")

        if img.max() > 0:
            img = (img - np.min(img)) / np.ptp(img)

        return img

    """
    Data loader calls this function to get an image as needed during training
    ================== ===========================================================================
    **Arguments:**
    index              dataframe index

    **Returns:**
    img                image of type tensor
    label              class labels of type tensor
    ================== ===========================================================================
    """
    def __getitem__(self, index):
        # (n_samples, channels, height, width)

        # the dataframe only has one image type for each subject.
        # we need to load all image types
        imgs = []

        for sequence in self.sequences:
            img = self.get_image(sequence, index)

            imgs.append(img)

        # image shape is (channels,height,width)
        imgs = np.asarray(imgs, dtype="float64")

        # add depth dimension so image shape is now (channels, height, width, depth) (needed for torchio)
        imgs = imgs[..., np.newaxis]

        # apply data augmentation
        img = self.transform(imgs)

        # remove depth dimension for training
        img = np.squeeze(img, axis=3)

        # convert numpy array to tensor
        img = torch.Tensor(img)

        # onehot encoding
        label = np.argmax(self.data_df.loc[index, self.target_cols])

        label = torch.tensor(label)

        return img, label

    """
    Returns the pytorch transform. The key transform is ToTensor() as the images 
    (type ndarrays) need to be converted to tensors before they can be input into 
    the model for training. We'll also standardize the dataframe to mean 0 and stdev of 1
    ================== ===========================================================================
    **Arguments:**
    data_df            labels dataframe

    **Returns:**
    _transform         pytorch transform
    ================== ===========================================================================
    """
    def get_transform(self):
        if self.augmentation:
            _transforms = tio.Compose([
                tio.ZNormalization(masking_method=self.get_foreground),
                tio.RandomNoise(p=0.5),  # Gaussian noise 50% of times
                tio.OneOf(
                    {  # either
                        tio.RandomAffine(): 1.0,  # random affine
                    }, p=0.8),  # applied to 80% of images
            ])

            # _transforms = tio.Compose([
            #     tio.ZNormalization(masking_method=self.get_foreground),
            #     tio.RandomBlur(p=0.5),  # blur 50% of times
            #     tio.RandomNoise(p=0.5),  # Gaussian noise 50% of times
            #     tio.OneOf({  # either
            #         tio.RandomAffine(): 0.75,  # random affine
            #         tio.RandomElasticDeformation(): 0.25,  # or random elastic deformation
            #     }, p=0.8),  # applied to 80% of images
            #     tio.OneOf({  # either
            #         tio.RandomBiasField(): 0.4,  # magnetic field inhomogeneity 40% * 0.8 = 32% of times
            #         tio.RandomMotion(): 0.2,  # or random motion artifact
            #         tio.RandomSpike(): 0.2,  # or spikes
            #         tio.RandomGhosting(): 0.2,  # or ghosts
            #     }, p=0.8),  # applied to 80% of images
            # ])

        else:
            _transforms = tio.Compose([
                tio.ZNormalization(masking_method=self.get_foreground),
            ])

        return _transforms

    def get_foreground(self, x):
        return x > x.mean()
