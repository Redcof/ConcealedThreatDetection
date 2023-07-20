import ast
import gc
import random
import time

import torch
import torchvision
from torch import optim
from torch.backends import cudnn
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

from src.config.config import read, classnames


class FasterRCNN_ResNet50_FPN_Trainer:
    def __init__(self, cfg):
        """
        Returns:
        https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
        #torchvision.models.detection.fasterrcnn_resnet50_fpn
        """
        self.cfg = cfg
        self.lr_scheduler = None
        manual_seed = self.cfg.random_seed
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        cudnn.benchmark = True
        self.net = self.build_model()
        self.optimizer = self.build_optimizer()
    
    def build_model(self):
        if self.cfg.model.name == "fasterrcnn_resnet50_fpn":
            if self.cfg.model.weights == "DEFAULT":
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            else:
                weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                         weights=weights)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.cfg.dataset.num_classes)
            return model
        else:
            raise NotImplemented(self.cfg.model.name)
    
    def build_optimizer(self):
        return optim.RMSprop(self.net.parameters(), lr=self.cfg.train.lr)
    
    def build_lr_scheduler(self):
        lr_schedule_dict = self.cfg.train.lr_schedule
        if lr_schedule_dict['name'] == "LinearLR":
            kwargs = {k: v for k, v in lr_schedule_dict.items() if k != "name"}
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, **kwargs
            )
        else:
            raise NotImplemented(self.cfg.train.lr_schedule)
    
    def train_one_epoch(self, train_dataloader, epoch_idx, max_epoch):
        losses = torch.tensor(-1)
        self.net.train()
        for batch_idx, (list_inputs, list_targets) in enumerate(train_dataloader):
            # Forward pass with losses
            loss_dict = self.net(list_inputs, list_targets)
            # sum classification losses + bbox regression losses
            losses = sum(v for v in loss_dict.values())
            # set gradient to zero
            # Calculate gradient
            self.optimizer.zero_grad()
            losses.backward()
            # update parameters
            self.optimizer.step()
            print("\rEpoch [%d/%d] Batch [%d/%d] Loss [%0.4e] " % (
                epoch_idx, max_epoch, batch_idx + 1, len(train_dataloader), losses), end="\b")
            
            # CLEAN GPU RAM  ########################
            del list_inputs
            del list_targets
            del loss_dict
            # Fix: https://discuss.pytorch.org/t/how-to-totally-free-allocate-memory-in-cuda/79590
            torch.cuda.empty_cache()
            gc.collect()
            # print("After memory_allocated(GB): ", torch.cuda.memory_allocated() / 1e9)
            # print("After memory_cached(GB): ", torch.cuda.memory_reserved() / 1e9)
            # CLEAN GPU RAM ########################
        return losses
    
    def train(self, train_dataloader, test_dataloader):
        self.net.to(self.cfg.device)
        self.net.train()
        print("Training...")
        for epoch_idx in range(self.cfg.train.start_epoch, self.cfg.train.max_epoch + 1, 1):
            if epoch_idx == self.cfg.train.lr_schedule_warmup_epoch:
                self.lr_scheduler = self.build_lr_scheduler()
            losses = self.train_one_epoch(train_dataloader, epoch_idx, self.cfg.train.max_epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            mAP_values = self.test(test_dataloader)
            print("\nEpoch %d Loss: %0.4e mAP:%0.4f" % (
                epoch_idx, losses,
                mAP_values['map']
            ))
        return self.net
    
    def build_metrics(self):
        metrics = ast.literal_eval(self.cfg.train.metrics)
    
    @torch.no_grad()
    def test(self, test_dataloader):
        # https://github.com/haochen23/Faster-RCNN-fine-tune-PyTorch/blob/master/engine.py#L69
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.net.eval()
        map_metric = MeanAveragePrecision()
        torch.cuda.synchronize()
        model_time = 0
        for batch_idx, (image, targets) in enumerate(test_dataloader):
            start_time = time.time()
            predictions = self.net(image)
            delta = time.time() - start_time
            model_time += delta
            
            for prediction, ground_truth in zip(predictions, targets):
                preds = [{
                    "boxes": torch.as_tensor(prediction["boxes"]).to(cpu_device),
                    "scores": torch.as_tensor(prediction["scores"]).to(cpu_device),
                    "labels": torch.as_tensor(prediction["labels"]).to(cpu_device),
                }]
                target = [{
                    "boxes": torch.as_tensor(ground_truth["boxes"]).to(cpu_device),
                    "labels": torch.as_tensor(ground_truth["labels"]).to(cpu_device),
                }]
                map_metric.update(preds, target)
        
        evaluator_time = time.time()
        mAP_values = map_metric.compute()
        evaluator_time = time.time() - evaluator_time
        # metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
        # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)
        # coco_evaluator.synchronize_between_processes()
        
        # accumulate predictions from all images
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        return mAP_values
