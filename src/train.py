from torch.utils.data import DataLoader

from config.config import read
from dataloader import ATZDataset
from pytorch_object_detection import FasterRCNN_ResNet50_FPN_Trainer


def dataloader_collate_fn(device):
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


def get_dataloader(cfg, split, collate_fn_wrap=dataloader_collate_fn):
    drop_last = True
    data_dir = cfg.dataset.data_root
    dataset = ATZDataset(data_dir, split, cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.train.batch_size,
                            drop_last=drop_last,
                            collate_fn=collate_fn_wrap(cfg.device),
                            shuffle=True)
    return dataloader


@read('./src/config/atz.yaml')
def main(cfg):
    if cfg.framework == "pytorch":
        model_trainer = FasterRCNN_ResNet50_FPN_Trainer(cfg)
    else:
        raise NotImplemented("Framework: %s" % cfg.framework)
    if cfg.train.flag:
        train_dataloader = None
        test_dataloader = None
        val_dataloader = None
        if cfg.dataset.voc_data:
            train_dataloader = get_dataloader(cfg, 'train')
            test_dataloader = get_dataloader(cfg, 'test')
            val_dataloader = get_dataloader(cfg, 'val')
        else:
            assert cfg.dataset.voc_data, "Only VOC dataset in implemented"
        model_trainer.train(train_dataloader, test_dataloader)
        # model_trainer.finetune(val_dataloader)


if __name__ == '__main__':
    main()
