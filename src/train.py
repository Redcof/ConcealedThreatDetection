import argparse

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from config.config import read
from dataset import ATZDataset
from pytorch_object_detection import PytorchDetectionTrainer


def dataloader_collate_fn_fasterrcnn(device):
    # prepare for FasterRCNN_ResNet50_FPN model
    def collate_wrapper(data):
        """
        This function is called in between your batch loop and dataloader
        """
        list_inputs = []
        list_targets = []
        for image, (boxes, labels) in data:
            list_inputs.append(image.to(device))
            list_targets.append({
                "boxes": boxes.to(device),
                "labels": labels.to(device),
            })
        return list_inputs, list_targets
    
    return collate_wrapper


def dataloader_collate_fn(device):
    # prepare for except FasterRCNN_ResNet50_FPN model
    def collate_wrapper(data):
        """
        This function is called in between your batch loop and dataloader
        """
        list_inputs = []
        list_bbox = []
        list_classes = []
        for image, (boxes, labels) in data:
            repeat = len(labels) - 1
            for b, l in zip(boxes, labels):
                list_classes.append(l.to(device))
                list_bbox.append(b.to(device))
            list_inputs.append(image.to(device))
            for i in range(repeat):
                list_inputs.append(image.to(device))
        return list_inputs, (list_bbox, list_classes)
    
    return collate_wrapper


def normalize(im):
    # https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0]) / imagenet_stats[1]


def get_dataloader(cfg, split, collate_fn_wrap):
    drop_last = True
    data_dir = cfg.dataset.data_root
    dataset = ATZDataset(data_dir, split, cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.train.batch_size,
                            drop_last=drop_last,
                            collate_fn=collate_fn_wrap(cfg.device),
                            shuffle=True)
    return dataloader


def main(cfg):
    if cfg.framework == "pytorch":
        model_trainer = PytorchDetectionTrainer(cfg)
    else:
        raise NotImplemented("Framework: %s" % cfg.framework)
    if cfg.train.flag:
        train_dataloader = None
        test_dataloader = None
        val_dataloader = None
        if cfg.dataset.voc_data:
            if cfg.model.name == "fasterrcnn_resnet50_fpn":
                collate_fn_wrap = dataloader_collate_fn_fasterrcnn
            else:
                collate_fn_wrap = dataloader_collate_fn
            train_dataloader = get_dataloader(cfg, 'train', collate_fn_wrap=collate_fn_wrap)
            test_dataloader = get_dataloader(cfg, 'test', collate_fn_wrap=collate_fn_wrap)
            combined_train_dataset = torch.utils.data.ConcatDataset([train_dataloader.dataset, test_dataloader.dataset])
            combined_dataloader = DataLoader(dataset=combined_train_dataset, batch_size=cfg.train.batch_size,
                                             drop_last=True,
                                             collate_fn=collate_fn_wrap(cfg.device),
                                             shuffle=True)
            val_dataloader = get_dataloader(cfg, 'val', collate_fn_wrap=collate_fn_wrap)
        else:
            assert cfg.dataset.voc_data, "Only VOC dataset in implemented"
        model_trainer.train(combined_dataloader, test_dataloader)
        # model_trainer.finetune(val_dataloader)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detection network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./src/config/atz.yaml', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    
    @read(args.cfg_file)
    def config(cfg):
        return cfg
    
    
    main(config())
