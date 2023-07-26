import os.path
import pathlib
from xml.etree import ElementTree

import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset


class ATZDataset(Dataset):
    """
    This is a wrapper for VOCDetection
    https://pytorch.org/vision/main/generated/torchvision.datasets.VOCDetection.html#vocdetection.
    to provide Active Terahertz Imaging Dataset for Concealed Object Detection(https://github.com/LingLIx/THz_Dataset)
    """
    
    def __init__(self, root: str, split: str, cfg, image_transform=None):
        assert split in ("train", "test", "val")
        self.data_dir = pathlib.Path(root)
        self.image_dir = self.data_dir / "JPEGImages"
        self.anno_dir = self.data_dir / "Annotations"
        self.collection_file = self.data_dir / ("%s.txt" % split)
        self.cfg = cfg
        assert os.path.isdir(self.data_dir)
        assert os.path.isdir(self.image_dir)
        assert os.path.isdir(self.anno_dir)
        assert os.path.isfile(self.collection_file)
        # load filenames form split files
        with open(self.collection_file) as fp:
            # skip files with no annotation
            self.filenames = list(filter(lambda fnm: len(self.get_objects(self.anno_dir / ("%s.xml" % fnm))) > 0,
                                         map(lambda x: x.strip(), fp.readlines())))
    
    def get_objects(self, xml_file: str):
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        # Skip Object of Interest for IGNORE classes
        OOI = list(filter(lambda _box: _box.find("name").text in self.cfg.dataset.classnames, root.iter('object')))
        return OOI
    
    def read_vocxml_content(self, xml_file: str):
        # Skip HUMAN bboxes
        OOI = self.get_objects(xml_file)
        N = len(OOI)
        assert N > 0, "At-least 1 object is required"
        boxes = torch.zeros((N, 4), dtype=torch.float32)
        labels = torch.zeros((N,), dtype=torch.int64)
        for idx, box in enumerate(OOI):
            class_ = box.find("name").text
            ymin = int(box.find("bndbox/ymin").text)
            xmin = int(box.find("bndbox/xmin").text)
            ymax = int(box.find("bndbox/ymax").text)
            xmax = int(box.find("bndbox/xmax").text)
            boxes[idx, :] = torch.tensor((xmin, ymin, xmax, ymax), dtype=torch.float32)
            labels[idx] = self.cfg.dataset.classnames.index(class_)
        return boxes, labels
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = torchvision.transforms.ToTensor()(Image.open(self.image_dir / ("%s.jpg" % filename)).convert("RGB"))
        targets = self.read_vocxml_content(self.anno_dir / ("%s.xml" % filename))
        return image, targets
