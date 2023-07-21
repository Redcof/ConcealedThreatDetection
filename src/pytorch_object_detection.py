import ast
import gc
import os
import random
import time

import mlflow
import pandas as pd
import torch
import torchvision
from torch import optim
from torch.backends import cudnn
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights


class FasterRCNN_ResNet50_FPN_Trainer:
    def __init__(self, cfg):
        """
        Returns:
        https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
        #torchvision.models.detection.fasterrcnn_resnet50_fpn
        """
        self.mlflow_model_io_signature = None
        self.cfg = cfg
        self.set_randomness()
        self.lr_scheduler = None
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
            raise NotImplemented("Model: %s" % self.cfg.model.name)
    
    def build_optimizer(self):
        if self.cfg.train.optimizer.name == "RMSprop":
            return optim.RMSprop(self.net.parameters(), lr=self.cfg.train.lr)
    
    def build_lr_scheduler(self):
        lr_schedule_dict = self.cfg.train.lr_schedule
        if lr_schedule_dict['name'] == "LinearLR":
            kwargs = {k: v for k, v in lr_schedule_dict.items() if k != "name"}
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, **kwargs
            )
        else:
            raise NotImplemented("LR Schedule: %s" % self.cfg.train.lr_schedule.name)
    
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
            print("\rTraining... Epoch [%d/%d] Batch [%d/%d] Loss [%0.4e] " % (
                epoch_idx, max_epoch, batch_idx + 1, len(train_dataloader), losses), end="\b")
            # mlflow experiment tracking
            mlflow.log_metrics(dict(step_loss=losses), step=((epoch_idx - 1) * self.cfg.train.batch_size) + batch_idx)
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
        # mlflow experiment tracking
        mlflow.log_metrics(dict(epoch_loss=losses), step=epoch_idx)
        return losses
    
    def set_randomness(self):
        manual_seed = self.cfg.random_seed
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        cudnn.benchmark = True
    
    def start_tracking(self):
        keys = [
            "MLFLOW_TRACKING_URI",
            "DAGSHUB_REPO",
            "DAGSHUB_USERNAME",
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD",
        ]
        for key in keys:
            assert os.environ[key] != "", "'%s' is required to set in environment" % keys
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        mlflow.set_experiment(experiment_name=self.cfg.exp_name)
        # mlflow.autolog(log_models=True)
        # dagshub.init(os.environ['DAGSHUB_REPO'], os.environ['DAGSHUB_USERNAME'], mlflow=True)
        mlflow.start_run()
        run = mlflow.active_run()
        print("Active mlflow run_id: {} started...".format(run.info.run_id))
    
    def stop_tracking(self):
        run = mlflow.active_run()
        mlflow.end_run()
        print("Active mlflow run_id: {} stopped.".format(run.info.run_id))
    
    def track_config(self):
        d = dict(self.cfg)
        del d['dataset']
        del d['model']
        del d['train']
        del d['inference']
        mlflow.log_params(d)
        mlflow.log_params(dict(
            dataset=self.cfg.dataset.name,
            voc_data=self.cfg.dataset.voc_data,
            algorithm=self.cfg.model.name,
            weights=self.cfg.model.weights,
            batch_size=self.cfg.train.batch_size,
            optimizer=self.cfg.train.optimizer.name,
            optimizer_param=self.cfg.train.optimizer,
            lr=self.cfg.train.lr,
            lr_schedule=self.cfg.train.lr_schedule.name,
            lr_schedule_warmup_epoch=self.cfg.train.lr_schedule_warmup_epoch,
            lr_schedule_param=self.cfg.train.lr_schedule,
            start_epoch=self.cfg.train.start_epoch,
            max_epoch=self.cfg.train.max_epoch,
            finetune=self.cfg.train.finetune.flag,
            checkpoint_path=self.cfg.train.finetune.checkpoint_path,
        ))
        d = {}
        if self.cfg.train.flag is True:
            d['Mode'] = "Training"
        elif self.cfg.inference.flag is True:
            d['Mode'] = "Inference"
        
        mlflow.set_experiment_tags(d)
    
    def train(self, train_dataloader, test_dataloader):
        self.start_tracking()
        self.track_config()
        self.net.to(self.cfg.device)
        self.net.train()
        print("Entering training loop...")
        for epoch_idx in range(self.cfg.train.start_epoch, self.cfg.train.max_epoch + 1, 1):
            if epoch_idx == self.cfg.train.lr_schedule_warmup_epoch:
                self.lr_scheduler = self.build_lr_scheduler()
            # train one epoch
            losses = self.train_one_epoch(train_dataloader, epoch_idx, self.cfg.train.max_epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # calculate mAP
            mAP_values = self.evaluate(test_dataloader, epoch_idx)
            print("\nTraining Epoch %d Loss: %0.4e mAP:%0.4f" % (
                epoch_idx, losses,
                mAP_values['map']
            ))
        # mlflow track model
        if self.mlflow_model_io_signature is not None and self.cfg.framework == "pytorch":
            print("Logging model to mlflow...", end="")
            mlflow.pytorch.log_model(self.net, "output/model", signature=self.mlflow_model_io_signature)
            print("Done")
        self.stop_tracking()
        return self.net
    
    def build_metrics(self):
        metrics = ast.literal_eval(self.cfg.train.metrics)
    
    def track_model_signature(self, image, prediction):
        if self.mlflow_model_io_signature is None:
            # this requires numpy ndarray or pandas dataframe datatype
            # as our input is a tensor we can freely convert it to numpy-array
            # as our out is a dictionary we can freely convert it to pandas dataframe
            to_cpu_numpy = lambda x: x.detach().cpu().numpy()
            sample_output = {
                "boxes": [float(1), float(1), float(1), float(1)],
                "scores": float(1),
                "labels": float(1),
            }
            self.mlflow_model_io_signature = mlflow.models.infer_signature(to_cpu_numpy(image),
                                                                           pd.DataFrame(sample_output))
    
    @torch.no_grad()
    def evaluate(self, test_dataloader, epoch_idx):
        # https://github.com/haochen23/Faster-RCNN-fine-tune-PyTorch/blob/master/engine.py#L69
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.net.eval()
        map_metric = MeanAveragePrecision()
        torch.cuda.synchronize()
        model_time = 0
        for batch_idx, (list_image, list_targets) in enumerate(test_dataloader):
            print("\rEvaluating... Epoch %d Batch [%d/%d] " % (
                epoch_idx, batch_idx + 1, len(test_dataloader)), end="\b")
            start_time = time.time()
            list_prediction = self.net(list_image)
            delta = time.time() - start_time
            # mlflow track experiment
            self.track_model_signature(list_image[0], list_prediction[0])
            # mlflow track experiment
            mlflow.log_metrics(dict(step_inference_time=model_time),
                               step=((epoch_idx - 1) * self.cfg.train.batch_size) + batch_idx)
            model_time += delta
            
            for prediction, ground_truth in zip(list_prediction, list_targets):
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
        # mlflow track experiment
        mlflow.log_metrics(dict(evaluator_time=evaluator_time), step=epoch_idx)
        mlflow.log_metrics(dict(epoch_inference_time=model_time), step=epoch_idx)
        mlflow.log_metrics(mAP_values, step=epoch_idx)
        torch.set_num_threads(n_threads)
        return mAP_values
