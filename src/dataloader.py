import os.path
import pathlib
from xml.etree import ElementTree

from PIL import Image
from torch.utils.data import Dataset


class ATZDataset(Dataset):
    """
    This is a wrapper for VOCDetection
    https://pytorch.org/vision/main/generated/torchvision.datasets.VOCDetection.html#vocdetection.
    to provide Active Terahertz Imaging Dataset for Concealed Object Detection(https://github.com/LingLIx/THz_Dataset)
    """
    
    def __init__(self, root: str, split: str, image_transform=None, label_transform=None, bbox_transform=None):
        assert split in ("train", "test", "val")
        self.data_dir = pathlib.Path(root)
        self.image_dir = self.data_dir / "JPEGImages"
        self.anno_dir = self.data_dir / "Annotation"
        self.collection_file = self.data_dir / ("%s.txt" % split)
        assert os.path.isdir(self.data_dir)
        assert os.path.isdir(self.image_dir)
        assert os.path.isdir(self.anno_dir)
        assert os.path.isfile(self.collection_file)
        # load filenames form split files
        with open(self.collection_file) as fp:
            self.filenames = list(map(lambda x: x.strip(), fp.readlines()))
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.bbox_transform = bbox_transform
    
    @staticmethod
    def read_vocxml_content(xml_file: str):
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        
        list_with_all_boxes = []
        
        filename = root.find('filename').text
        for boxes in root.iter('object'):
            class_ = boxes.find("name").text
            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            
            list_with_single_boxes = (xmin, ymin, xmax, ymax, cx, cy, class_)
            list_with_all_boxes.append(list_with_single_boxes)
        
        return filename, list_with_all_boxes
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(self.image_dir / ("%s.jpeg" % filename)).convert("RGB")
        bboxes = self.read_vocxml_content(self.anno_dir / ("%s.xml" % filename))
