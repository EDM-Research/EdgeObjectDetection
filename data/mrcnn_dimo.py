import random

from mrcnn import utils, visualize, config
from data.dimo_loader import DimoLoader
from pathlib import Path
from typing import List, Tuple
import os
import numpy as np
import skimage


class DimoConfig(config.Config):
    NAME = "dimo"
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = True
    NUM_CLASSES = 8 + 1     # 8 models + background
    STEPS_PER_EPOCH = 1000
    TRAIN_ROIS_PER_IMAGE = 50
    LEARNING_RATE = 0.0001


class DimoInferenceConfig(config.Config):
    NAME = "dimo"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 8 + 1     # 8 models + background
    DETECTION_MIN_CONFIDENCE = 0.5
    USE_MINI_MASK = False


class DIMODataset(utils.Dataset):
    def load_dataset(self, path: str, subsets: List[str], split: str = "train"):
        assert split in ["train", "val", "test"]
        self.split = split

        dimo_loader = DimoLoader()
        dimo_ds = dimo_loader.load(Path(path), cameras=subsets)
        image_no = 0
        debug = os.environ.get("DEBUG_MODE", "0") == "1"

        model_to_id = {}

        for i, model in enumerate(dimo_ds['models']):
            self.add_class("dimo", i + 1, str(model['id']))
            model_to_id[str(model['id'])] = i + 1

        for subset_name in subsets:
            subset = dimo_ds[subset_name]
            scene_ids = self.get_scene_ids(path, subset_name)
            for scene in subset:
                if scene['id'] not in scene_ids:
                    continue
                masks_path = os.path.join(scene['path'], 'mask_visib/')
                for image in scene['images']:
                    instance_masks = []
                    instance_ids = []
                    for i, object in enumerate(image['objects']):
                        instance_ids.append(model_to_id[str(object['id'])])
                        instance_masks.append(os.path.join(masks_path, f"{str(image['id']).zfill(6)}_{str(i).zfill(6)}.png"))

                    image_data = skimage.io.imread(image['path'])
                    height, width = image_data.shape[:2]

                    self.add_image(
                        "dimo",
                        image_id=f"{subset_name}_{scene['id']}_{image['id']}",
                        path=image['path'],
                        instance_masks=instance_masks,
                        instance_ids=instance_ids,
                        width=width,
                        height=height
                    )
                    image_no += 1

                    if debug and image_no > 20:
                        return

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        masks = np.zeros((image_info['height'], image_info['width'], len(image_info['instance_ids'])))
        for object_no, mask_path in enumerate(image_info["instance_masks"]):
            masks[:, :, object_no] = skimage.io.imread(mask_path)
        return masks.astype(np.uint8), np.array(image_info["instance_ids"], dtype=np.int32)

    def image_reference(self, image_id):
        return self.image_info[image_id]['path']

    def get_scene_ids(self, path: str, subset: str) -> List[int]:
        subset_path = os.path.join(path, subset)
        split_file_path = os.path.join(subset_path, f"{self.split}.txt")
        with open(split_file_path, 'r') as f:
            ids = [int(line.rstrip()) for line in f]
        return ids


def get_dimo_datasets(path: str, subsets: List[str]) -> Tuple[DIMODataset, DIMODataset]:
    dataset_train = DIMODataset()
    dataset_train.load_dataset(path, subsets, split="train")
    dataset_train.prepare()

    dataset_val = DIMODataset()
    dataset_val.load_dataset(path, subsets, split="val")
    dataset_val.prepare()
    return dataset_train, dataset_val


def get_test_dimo_dataset(path: str, subsets: List[str]) -> DIMODataset:
    dataset = DIMODataset()
    dataset.load_dataset(path, subsets, split="test")
    dataset.prepare()
    return dataset
