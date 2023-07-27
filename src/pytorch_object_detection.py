import ast
import gc
import os
import random
import time

import mlflow
import numpy as np
import torch
import torchvision
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from torch import optim, nn
from torch.backends import cudnn
from torchgen.context import F
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

from config.config import read_yaml


class PytorchDetectionTrainer:
    def __init__(self, cfg, imagedim):
        """
        Returns:
        https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
        #torchvision.models.detection.fasterrcnn_resnet50_fpn
        """
        self.last_value = None
        self.model_saved_earlier = False
        self.forward = None
        self.mlflow_model_io_signature = None
        self.cfg = cfg
        self.imagedim = imagedim
        self.set_randomness()
        self.lr_scheduler = None
        self.net = self.build_model()
        self.optimizer = self.build_optimizer()
    
    def build_model(self):
        model_name = self.cfg.model.name
        num_classes = self.cfg.dataset.num_classes
        if model_name == "fasterrcnn_resnet50_fpn":
            if self.cfg.model.weights == "DEFAULT":
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                pretrained = True
            elif self.cfg.model.weights == "COCO_V1":
                weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
                pretrained = True
            else:
                pretrained = False
                weights = None
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                         weights=weights)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            def forward(list_input, list_target):
                # forward pass for FasterRCNN model
                loss_dict = model(list_input, list_target)
                return sum(v for v in loss_dict.values())
            
            # this requires numpy ndarray or pandas dataframe datatype
            # as our input is a tensor we can freely convert it to numpy-array
            # as our out is a dictionary we can freely convert it to pandas dataframe
            input_schema = Schema(
                [TensorSpec(np.dtype(np.float32), (-1, 880, 335))]
            )
            output_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, 4), "boxes"),
                TensorSpec(np.dtype(np.float32), (-1, 1), "scores"),
                TensorSpec(np.dtype(np.int64), (-1, 1), "labels"),
            ])
            self.mlflow_model_io_signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        elif model_name == "custom_resnet":
            class DetectionModel(nn.Module):
                def __init__(self):
                    super(DetectionModel, self).__init__()
                    from torchvision.models import ResNet34_Weights
                    resnet = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
                    layers = list(resnet.children())[:8]
                    self.features1 = nn.Sequential(*layers[:6])
                    self.features2 = nn.Sequential(*layers[6:])
                    self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
                    self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
                
                def forward(self, x):
                    x = self.features1(x)
                    x = self.features2(x)
                    x = F.relu(x)
                    x = nn.AdaptiveAvgPool2d((1, 1))(x)
                    x = x.view(x.shape[0], -1)
                    return self.classifier(x), self.bb(x)
            
            model = DetectionModel()
            
            def forward(batch_input, batch_target):
                y_bb, y_class = batch_target
                # forward pass
                out_cls, out_bbox = model(batch_input)
                # calculate class loss
                loss_class = F.cross_entropy(out_cls, y_class, reduction="sum")
                # calculate bbox loss
                loss_bb = F.l1_loss(out_bbox, y_bb, reduction="none").sum(1)
                loss_bb = loss_bb.sum()
                loss = loss_class + loss_bb / num_classes
                return loss
            
            # this requires numpy ndarray or pandas dataframe datatype
            # as our input is a tensor we can freely convert it to numpy-array
            # as our out is a dictionary we can freely convert it to pandas dataframe
            input_schema = Schema(
                [TensorSpec(np.dtype(np.float32), (-1, 880, 335))]
            )
            output_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, 4), "boxes"),
                TensorSpec(np.dtype(np.float32), (-1, 1), "scores"),
                TensorSpec(np.dtype(np.int64), (-1, 1), "labels"),
            ])
            self.mlflow_model_io_signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        else:
            raise NotImplemented("Model: %s" % self.cfg.model.name)
        self.forward = forward
        return model
    
    def build_optimizer(self):
        if self.cfg.train.optimizer.name == "RMSprop":
            return optim.RMSprop(self.net.parameters(), lr=self.cfg.train.lr)
        elif self.cfg.train.optimizer.name == "Adam":
            return optim.Adam(self.net.parameters(), lr=self.cfg.train.lr)
    
    def build_lr_scheduler(self):
        if self.cfg.train.lr_schedule.flag:
            lr_schedule_dict = self.cfg.train.lr_schedule
            if lr_schedule_dict['name'] == "LinearLR":
                kwargs = {k: v for k, v in lr_schedule_dict.items() if k != "name"}
                return torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, **kwargs
                )
            else:
                raise NotImplemented("LR Schedule: %s" % self.cfg.train.lr_schedule.name)
    
    @read_yaml('./earlystop.yaml')
    def is_early_stop(self, cfg, epoch_idx=-1):
        if cfg.hard_stop_now:
            print("Early stopping...", end="")
            print("HARD STOP.")
            return True
        if cfg.stop_at_epoch == epoch_idx and cfg.stop_at_epoch != -1:
            print("Early stopping...", end="")
            print("EPOCH STOP.")
            return True
    
    def train_one_epoch(self, train_dataloader, epoch_idx, max_epoch):
        loss = torch.tensor(-1)
        self.net.train()
        for batch_idx, (batch_inputs, batch_targets) in enumerate(train_dataloader):
            # Forward pass with loss
            # output = self.net(batch_inputs, batch_targets)
            # sum classification loss + bbox regression loss
            loss = self.forward(batch_inputs, batch_targets)
            # set gradient to zero
            # Calculate gradient
            self.optimizer.zero_grad()
            loss.backward()
            # update parameters
            self.optimizer.step()
            print("\rTraining... Epoch [%d/%d] Batch [%d/%d] Loss [%0.4e] " % (
                epoch_idx, max_epoch, batch_idx + 1, len(train_dataloader), loss), end="\b")
            # mlflow experiment tracking
            mlflow.log_metrics(dict(step_loss=loss.item()),
                               step=((epoch_idx - 1) * self.cfg.train.batch_size) + batch_idx)
            # CLEAN GPU RAM  ########################
            del batch_inputs
            del batch_targets
            # Fix: https://discuss.pytorch.org/t/how-to-totally-free-allocate-memory-in-cuda/79590
            torch.cuda.empty_cache()
            gc.collect()
            # print("After memory_allocated(GB): ", torch.cuda.memory_allocated() / 1e9)
            # print("After memory_cached(GB): ", torch.cuda.memory_reserved() / 1e9)
            # CLEAN GPU RAM #
            if self.is_early_stop(epoch_idx=epoch_idx):
                break
            # ONE EPOCH LOOP ENDS #######################
        # mlflow experiment tracking
        mlflow.log_metrics(dict(epoch_loss=loss.item()), step=epoch_idx)
        return loss
    
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
        mlflow.start_run(description=self.cfg.experiment_description)
        run = mlflow.active_run()
        # track additional hparmas
        self.track_config()
        print("Active mlflow run_name:{} run_id: {} started...".format(run.info.run_name, run.info.run_id))
    
    def stop_tracking(self):
        run = mlflow.active_run()
        mlflow.end_run()
        print("Active mlflow run_name:{} run_id: {} started...".format(run.info.run_name, run.info.run_id))
    
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
        # mlflow start tracking
        self.start_tracking()
        
        # track data stat
        mlflow.log_params(dict(
            train_datasize=len(train_dataloader.dataset),
            train_batch_size=len(train_dataloader),
            test_datasize=len(test_dataloader.dataset),
            test_batch_size=len(test_dataloader),
        ))
        self.net.to(self.cfg.device)
        self.net.train()
        print("Entering training loop...")
        for epoch_idx in range(self.cfg.train.start_epoch, self.cfg.train.max_epoch + 1, 1):
            if epoch_idx == self.cfg.train.lr_schedule_warmup_epoch:
                self.lr_scheduler = self.build_lr_scheduler()
            
            # train one epoch
            epoch_time = time.time()
            losses = self.train_one_epoch(train_dataloader, epoch_idx, self.cfg.train.max_epoch)
            epoch_time = time.time() - epoch_time
            
            # update LR
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # calculate mAP
            mAP_train_values = self.evaluate(train_dataloader, epoch_idx, "train")
            mAP_test_values = self.evaluate(test_dataloader, epoch_idx, "test")
            
            # mlflow track progress
            progress = ((epoch_idx / (self.cfg.train.max_epoch + 1)) * 100)
            mlflow.log_metrics(dict(epoch_time=epoch_time), step=epoch_idx)
            print(
                "\nTraining Progress: %0.2f%% Epoch [%d/%d] Loss: %0.4e Train map@IoU0.75:%0.4f Test map@IoU0.75:%0.4f" % (
                    progress,
                    epoch_idx, self.cfg.train.max_epoch + 1, losses,
                    mAP_train_values['train_map_75'],
                    mAP_test_values['test_map_75']
                ))
            
            # mlflow track model
            if self.can_i_log_model(epoch_idx, losses, mAP_train_values, mAP_test_values):
                self.log_model("epoch_%d" % epoch_idx)
            if self.is_early_stop(epoch_idx=epoch_idx):
                break
            # TRAIN LOOP ENDS ##############
        if not self.model_saved_earlier:
            self.log_model("epoch_last")
        # mlflow stop tracking
        self.stop_tracking()
        return self.net
    
    def can_i_log_model(self, epoch_idx, loss, train_performance, test_performance):
        schedule_type = self.cfg.train.save_schedule.type
        schedule_key = self.cfg.train.save_schedule.key
        schedule_delta = self.cfg.train.save_schedule.value_delta
        threshold = self.cfg.train.save_schedule.threshold
        flag = False
        if schedule_type == "epoch":
            if epoch_idx % schedule_key == 0:
                self.model_saved_earlier = True
                flag = True
            else:
                flag = False
        elif schedule_type == "metric":
            # get the current value
            current_value = train_performance[schedule_key] if (train_performance
                                                                and schedule_key in train_performance) else (
                test_performance[schedule_key] if (
                    test_performance and schedule_key in test_performance) else None
            )
            if current_value is not None:
                if self.last_value is None:
                    # Last value none indicated the training just started.
                    # set the last_value to current_value
                    self.last_value = current_value
                if (current_value >= threshold
                    and current_value > self.last_value
                    and abs(current_value - self.last_value) >= schedule_delta):
                    self.model_saved_earlier = True
                    flag = True
        elif schedule_type == "loss":
            current_value = loss.item() if loss is not None else None
            if current_value is None:
                flag = False
            else:
                if self.last_value is None:
                    # Last value none indicated the training just started.
                    # set the last_value to current_value
                    self.last_value = current_value
                if (current_value <= threshold
                    and current_value < self.last_value
                    and abs(current_value - self.last_value) >= schedule_delta):
                    self.model_saved_earlier = True
                    flag = True
        else:
            # do nothing
            ...
        return flag
    
    def log_model(self, dir_name):
        if self.mlflow_model_io_signature is not None and self.cfg.framework == "pytorch":
            print("Logging eager-model to mlflow backend...", end="")
            mlflow.pytorch.log_model(self.net, "output/%s/model" % dir_name,
                                     signature=self.mlflow_model_io_signature,
                                     pip_requirements="./requirements.txt")
            print("Done")
            print("Logging script-model to mlflow backend...", end="")
            mlflow.pytorch.log_model(torch.jit.script(self.net), "output/%s/scripted" % dir_name,
                                     signature=self.mlflow_model_io_signature,
                                     pip_requirements="./requirements.txt")
            print("Done")
            # torch.onnx.export(self.net, x, "faster_rcnn.onnx", opset_version=11)
    
    def build_metrics(self):
        metrics = ast.literal_eval(self.cfg.train.metrics)
    
    def track_model_signature(self, signature):
        if self.mlflow_model_io_signature is None:
            self.mlflow_model_io_signature = signature
    
    @torch.no_grad()
    def evaluate(self, test_dataloader, epoch_idx, prefix):
        # https://github.com/haochen23/Faster-RCNN-fine-tune-PyTorch/blob/master/engine.py#L69
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        map_metric = MeanAveragePrecision()
        torch.cuda.synchronize()
        model_time = 0
        # run evaluation loop
        self.net.eval()
        for batch_idx, (list_image, list_targets) in enumerate(test_dataloader):
            print("\rEvaluating...%s Epoch %d Batch [%d/%d] " % (
                prefix, epoch_idx, batch_idx + 1, len(test_dataloader)), end="\b")
            start_time = time.time()
            list_prediction = self.net(list_image)
            delta = time.time() - start_time
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
        
        # calculate mAP performance
        evaluator_time = time.time()
        mAP_values = map_metric.compute()
        evaluator_time = time.time() - evaluator_time
        mAP_values = {"%s_%s" % (prefix, key): value.item() for key, value in mAP_values.items()}
        # mlflow track experiment
        if prefix == "train":
            mlflow.log_metrics(dict(train_evaluator_time=evaluator_time), step=epoch_idx)
            mlflow.log_metrics(dict(train_epoch_inference_time=model_time), step=epoch_idx)
        if prefix == "test":
            mlflow.log_metrics(dict(test_evaluator_time=evaluator_time), step=epoch_idx)
            mlflow.log_metrics(dict(test_epoch_inference_time=model_time), step=epoch_idx)
        if prefix == "val":
            mlflow.log_metrics(dict(val_evaluator_time=evaluator_time), step=epoch_idx)
            mlflow.log_metrics(dict(val_epoch_inference_time=model_time), step=epoch_idx)
        mlflow.log_metrics(mAP_values, step=epoch_idx)
        torch.set_num_threads(n_threads)
        return mAP_values
