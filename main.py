import data.mrcnn_dimo
from data import utils as data_utils
from data import mrcnn_dimo
import os, random
from mrcnn import visualize as mrcnn_visualise
from mrcnn import model as modellib
from training import evaluation, detection
from utils import visualize, interactions, file_io
import configparser
import cv2
import numpy as np
from training import mrcnn as mrcnn_training
import tensorflow.keras.backend as K
from training.augmentation import EdgeAugment
import logging.config

config = configparser.ConfigParser()
config.read('config.ini')

DIMO_PATH = config['USER_SETTINGS']['dimo_path']


logfile = "log.txt"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

def test_batch(batch_file: str):
    model_tests = file_io.read_test_batch(batch_file)

    for test in model_tests:
        dataset = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, [test.test_subset])
        config = data.mrcnn_dimo.get_test_dimo_config(dataset, test.model_id)

        model = mrcnn_training.load_model(test.model_id, config)
        results = detection.get_detections_dataset(dataset, model, config)
        ap_50 = evaluation.compute_map(results, dataset, config, 0.5)
        ap_75 = evaluation.compute_map(results, dataset, config, 0.75)
        ap = evaluation.compute_coco_ap(results, dataset, config)

        test.metrics = {
            "ap": ap,
            "ap_50": ap_50,
            "ap_75": ap_75,
        }

    filename = f"{batch_file.split('.')[0]}_results.csv"
    file_io.write_test_metrics(model_tests, filename)


def train_subsets(subsets: list, model_id: str = None, augment: bool = False, transfer_learning: bool = False,
                  train_image_counts: list = None, ft_subsets: list = None, ft_image_count: int = None, layers: str = None, save_all: bool = False, dimo_path: str = None):
    target_dimo_path = dimo_path if dimo_path is not None else DIMO_PATH
    # load training set
    train, val, config = mrcnn_dimo.get_dimo_datasets(target_dimo_path, subsets, train_image_counts=train_image_counts)

    # if specified, load fintuning dataset
    ft_train = None
    if ft_subsets:
        ft_train, _, _ = mrcnn_dimo.get_dimo_datasets(target_dimo_path, ft_subsets, train_image_counts=ft_image_count)

    # load model to continue training, if specified
    model = mrcnn_training.load_model(model_id, config, mode="training") if model_id else None
    layers = layers if layers else 'heads'
    # train model
    mrcnn_training.train(train, val, config, checkpoint_model=model, ft_train_set=ft_train, layers=layers, save_all=save_all)


def prepare_subsets(subsets: list, override: bool = False, split_scenes: bool = False, dimo_path: str = None):
    target_dimo_path = dimo_path if dimo_path is not None else DIMO_PATH
    #data_utils.create_dimo_masks(target_dimo_path, subsets, override=override)
    data_utils.create_dimo_train_split(target_dimo_path, subsets, seed=10, split_scenes=split_scenes)


def show_subsets(subsets: list, dimo_path: str = None):
    target_dimo_path = dimo_path if dimo_path is not None else DIMO_PATH
    dataset_train, dataset_val, config = mrcnn_dimo.get_dimo_datasets(target_dimo_path, subsets)
    config.USE_MINI_MASK = False
    augmenter = EdgeAugment()

    print(f"training images: {len(dataset_train.image_ids)}")
    print(f"validation images: {len(dataset_val.image_ids)}")

    while True:
        image_id = random.choice(dataset_train.image_ids)
        image_info = dataset_train.image_info[image_id]
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_train, config, image_id)
        image = augmenter(image=image)
        mrcnn_visualise.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, title=image_info['id'], show_mask=False)


def test_subsets(subsets: list, model_id: str, save_results: bool = False, dimo_path: str = None):
    dataset = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, subsets)
    config = data.mrcnn_dimo.get_test_dimo_config(dataset, model_id)

    model = mrcnn_training.load_model(model_id, config)
    results = detection.get_detections_dataset(dataset, model, config)
    ap_50 = evaluation.compute_map(results, dataset, config, 0.5)
    ap_75 = evaluation.compute_map(results, dataset, config, 0.75)
    ap = evaluation.compute_coco_ap(results, dataset, config)

    print(f"AP = {ap}")
    print(f"AP 50 = {ap_50}")
    print(f"AP 75 = {ap_75}")

    if save_results:
        visualize.save_results(results, mrcnn_dimo.get_dataset_images(dataset, config), "results/", dataset.class_names)
    else:
        visualize.show_results(results, mrcnn_dimo.get_dataset_images(dataset, config), dataset.class_names)


def test_folder(folder: str,  model_id: str, num_classes: int, select_roi=False, save_folder=None):
    config = data.mrcnn_dimo.DimoInferenceConfig(num_classes=num_classes)
    model = mrcnn_training.load_model(model_id, config)

    images = [cv2.cvtColor(cv2.imread(os.path.join(folder, file)),cv2.COLOR_BGR2RGB) for file in os.listdir(folder) if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
    class_names = [str(i) for i in range(num_classes)]

    if select_roi:
        rois = interactions.select_rois(images)
        images = [image[r[0]:r[1], r[2]: r[3]] for r, image in zip(rois, images)]

    results = detection.get_detections_images(images, model)

    if save_folder:
        visualize.save_results(results, images, save_folder, class_names)
    else:
        visualize.show_results(results, images, class_names)


def test_epochs(subsets: list, models: list):
    test_frequency = 2
    dataset = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, subsets)

    for model_id in models:
        config = data.mrcnn_dimo.get_test_dimo_config(dataset, model_id)
        available_epochs = np.array(mrcnn_training.get_available_epochs(f"{mrcnn_training.get_model_folder()}/{model_id}"))
        test_epochs = np.arange(0, np.max(available_epochs) + 1, test_frequency).astype(np.int)
        test_epochs[0] += 1
        tested_epochs = []
        aps = []
        for epoch in test_epochs:
            if epoch in available_epochs:
                model = mrcnn_training.load_model(model_id, config, epoch)
                results = detection.get_detections_dataset(dataset, model, config)
                ap = evaluation.compute_coco_ap(results, dataset, config)

                tested_epochs.append(epoch)
                aps.append(ap)

                del model
                K.clear_session()

        file_io.write_model_epochs(model_id, aps, tested_epochs)


if __name__ == "__main__":
    pass
