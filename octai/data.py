import os
import numpy as np
import pandas as pd
import cv2

import pytorch_lightning as L

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder

from .transforms import ImageTransformer, apply_transforms
from .utils import prepare_config

# TO-DO:
# - add configurable class config

CLASS_NAMES = ('Norm', 'Early VLS', 'VLS')

class OctClassifierDataset(Dataset):

    default_config = {
        "transforms": None
    }
    
    def __init__(self, image_root, df_fp, tag, config = None):
        super(OctClassifierDataset, self).__init__()
        self.config = prepare_config(self, config)

        if (tag == "val") or (tag == "test"):
            self.config["transforms"]["p_augs"] = 0

        self.image_root = image_root
        self.df_fp = df_fp
        self.df = pd.read_csv(self.df_fp)
        
        self.labels = self.df.Condition.values
        self.labels = self._encode_labels(self.labels)
        paths = self.df["LOCAL_FILEPATH"]
        if paths.isna().any() or paths.eq("").any():
            raise ValueError("Image paths are blank in the public manifest; supply private relative paths before training or inference")
        self.image_paths = paths.apply(lambda local_fp: os.path.join(self.image_root, local_fp)).values
        
        self.transforms = ImageTransformer.get_transforms(config = self.config["transforms"])
    
    @staticmethod
    def _encode_labels(labels):
        encoder = OrdinalEncoder(categories = [CLASS_NAMES], dtype = int)

        return encoder.fit_transform(labels.reshape((-1,1))).flatten()
    
    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def _read_image(img_fp):
        image = cv2.imread(img_fp, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Cannot read OCT image: {img_fp}")
        return image
    
    @staticmethod
    def _transform_image(image, transforms):
        # translate image to 3 axes
        image = np.expand_dims(image, axis = 2)
        image = np.repeat(image, 3, axis = 2)
        # transform image
        image = apply_transforms(image, transforms)
        
        return image
    
    def __getitem__(self, idx):
        img_fp = self.image_paths[idx]
        image = self._read_image(img_fp)
        image = self._transform_image(image, self.transforms)
        image = image.transpose(2,0,1)

        label = self.labels[idx]
        
        return image, label


class MILOctClassifierDataset(OctClassifierDataset):
    
    def __init__(self, image_root, df_fp, tag, config = None):
        super().__init__(image_root, df_fp, tag, config = config)
        
        # RECODE LABELS
        self.labels_raw = self.df.groupby("measurement_id").Condition.apply(lambda df: df.iloc[0])
        self.measurement_ids = self.labels_raw.index.values
        self.labels = self._encode_labels(self.labels_raw.values)
        
        self.unique_slice_nums = self.df.slice_num.unique()
        self.unique_slice_nums_set = set(self.unique_slice_nums)
        self.slices_per_measurement = len(self.unique_slice_nums)
        
        # REINITIALIZE TRANSFORMS
        self.transforms = ImageTransformer.get_transforms(config = self.config['transforms'],
                                                             num_images = self.slices_per_measurement)
        
        
    def __len__(self):
        return self.labels.__len__()
    
    
    def get_measurement_df(self, measurement_id):
        measurement_df = self.df[self.df.measurement_id == measurement_id].set_index('slice_num', drop = False)
        
        # fill in missing slices with closest existing
        missing_slices = list(self.unique_slice_nums_set - set(measurement_df.index.values))
        for slice_num in missing_slices:
            closest_slice_num = measurement_df.index.values[np.abs(measurement_df.index.values - slice_num).argmin()]
            measurement_df.loc[slice_num] = measurement_df.loc[closest_slice_num]
            
        return measurement_df.sort_index()
        
    def __getitem__(self, idx):
        measurement_id = self.measurement_ids[idx]
        measurement_df = self.get_measurement_df(measurement_id)
        
        images = measurement_df.LOCAL_FILEPATH.apply(lambda path: self._read_image(os.path.join(
                                                self.image_root, path))[np.newaxis,:]).values
        # (num_slices X H X W)
        images = np.vstack(images)
        
        # (num_slices X H X W X 3)
        images = np.repeat(images[...,np.newaxis], 3, axis = 3)
        
        # apply transforms
        images_dict = self.transforms(image = images[0], **{f"image{i}":img for i,img in enumerate(images[1:])})
        images_transformed = np.stack([images_dict['image']] + [images_dict[f'image{i}']
                                                                for i in range(self.slices_per_measurement-1)],
                                                                          axis = 0)
        
        # shift color channel to front
        images_transformed = images_transformed.transpose(0, 3, 1, 2)

        label = self.labels[idx]
        
        return images_transformed, label
    
class VolumeOctClassifierDataset(MILOctClassifierDataset):
    def __getitem__(self, idx):
        images_transformed, label = super().__getitem__(idx)
        
        # leave single color channel
        images_transformed = images_transformed[:,[0],...]
        # permute depth channel to the second axis
        images_transformed = images_transformed.transpose(1,0,2,3)
        
        return images_transformed, label
    
class OctClassifierDataModule(L.LightningDataModule):

    default_config = {
        "batch_size": 64,
        "num_workers": 2,
        "dataset_type": "normal",
        "dataset": None
    }

    DATASETS = {
        "normal": OctClassifierDataset,
        "MIL": MILOctClassifierDataset,
        "3D": VolumeOctClassifierDataset
    }

    def __init__(self, image_root, train_csv, val_csv, test_csv, config = None, predict_on = 'test'):
        super(OctClassifierDataModule, self).__init__()

        self.config = prepare_config(self, config)
        
        self.image_root = image_root
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv

        self.predict_on = predict_on
        
    def setup(self, stage = None):
        DatasetClass = self.DATASETS[self.config['dataset_type']]

        self.train_dataset = DatasetClass(self.image_root, self.train_csv, tag = "train", config = self.config["dataset"])
        self.val_dataset = DatasetClass(self.image_root, self.val_csv, tag = "val", config = self.config["dataset"])
        self.test_dataset = DatasetClass(self.image_root, self.test_csv, tag = "test", config = self.config["dataset"])

        self.predict_dataset = self.__getattribute__(f"{self.predict_on}_dataset")
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.config["batch_size"], num_workers = self.config["num_workers"],
                         shuffle = True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.config["batch_size"], num_workers = self.config["num_workers"],
                         shuffle = False)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.config["batch_size"], num_workers = self.config["num_workers"],
                         shuffle = False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.config["batch_size"], num_workers = self.config["num_workers"],
                         shuffle = False)