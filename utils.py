import json
import pandas as pd
import numpy as np
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

import torch
from torch.nn import functional as F
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchmetrics import Accuracy, AUC, Precision, Recall, ConfusionMatrix
import pytorch_lightning as pl

from sklearn.model_selection import StratifiedKFold

def build_model(model_name, class_num, pretrained):
    print(f"build_model model_name:{model_name}, class_num:{class_num}, pretrained:{pretrained}")
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    else: # モデル名が不正のときはresnet18
        model = models.resnet18(pretrained=pretrained)

    # TrainingをFalseに
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, class_num)

    return model

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image paths. Extends
    torchvision.datasets.ImageFolder
    """
    
    # override the __getitem__ method that dataloader calls
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class LitModel(pl.LightningModule):
    def __init__(self, model, class_names, output_dir=None):
        super().__init__()
        self.model = model
        self.class_names = class_names
        self.output_dir = output_dir
        
        # average="none"でクラスごとにメトリクス計算
        num_classes = len(class_names)
        self.accuracy = Accuracy(num_classes=num_classes)
        self.prec = Precision(num_classes=num_classes) # 変数名をprecisionにすると何故かエラーになる
        self.recall = Recall(num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(num_classes=num_classes)
        self.test_paths = []
        self.test_true_label = []
        self.test_pred_label = []
        self.prob = []

    def forward(self, x):
        output = self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        return optimizer
        
    def training_step(self, batch, batch_idx):
        imgs, labels, paths = batch
        pred = self.model(imgs)
        # loss & accuracy
        prob = F.softmax(pred, dim=1)
        log = torch.logit(prob)
        loss = F.cross_entropy(log, labels, label_smoothing=0.1)

        # Log
        batch_size = len(imgs)
        self.log(f'train_loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'train_accuracy', self.accuracy(prob, labels), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'train_precision', self.prec(prob, labels), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'train_recall', self.recall(prob, labels), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, paths = batch
        pred = self.model(imgs)
        # loss & accuracy
        prob = F.softmax(pred, dim=1)
        log = torch.logit(prob)
        loss = F.cross_entropy(log, labels)

        # Log
        batch_size = len(imgs)
        self.log(f'val_loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'val_accuracy', self.accuracy(prob, labels), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'val_precision', self.prec(prob, labels), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'val_recall', self.recall(prob, labels), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        imgs, labels, paths = batch
        pred = self.model(imgs)
        prob = F.softmax(pred, dim=1)
        pred_label = pred.argmax(dim=1)

        # Log
        batch_size = len(imgs)
        self.log(f'test_accuracy', self.accuracy(prob, labels), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'test_precision', self.prec(prob, labels), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f'test_recall', self.recall(prob, labels), prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)

        return [paths, labels.tolist(), pred_label.tolist(), prob.tolist()]
    
    def test_epoch_end(self, outputs):
        # まとめて処理するためにnumpy arrayに変換する
        outputs = np.array(outputs, dtype=object)
        outputs = outputs[:,:,0]

        # prob以外をdfに
        result_df = pd.DataFrame(outputs[:,:-1], columns=["path", "true_label", "pred_label"])
        # prob(クラス確率のリスト)をクラスごとに列を作成し、dfに変換
        prob_df = pd.Series(outputs[:,-1]).apply(pd.Series)
        prob_df.columns = self.class_names
        # くっつけて保存
        result_df = pd.concat([result_df, prob_df], axis=1)
        result_df.to_csv(os.path.join(self.output_dir, "result.csv"), index=False)

        # class名のリストをdictに変換
        class_num = len(self.class_names)
        class_dict = {i: self.class_names[i] for i in range(class_num)}
        # ラベルのindexをラベル名に変換
        true_label = result_df["true_label"].replace(class_dict)
        pred_label = result_df["pred_label"].replace(class_dict)
        heatmap = pd.crosstab(true_label, pred_label)
        # 描画 & 保存
        rcParams['figure.figsize'] = 12,10
        sns.set(font_scale = 2.5)
        plt.title("Confusion Matrix")
        sns.heatmap(heatmap, annot=True, cmap="BuGn")
        plt.savefig(os.path.join(self.output_dir, "matrix.png"))

        return result_df


class JsonLogger(pl.loggers.base.LightningLoggerBase):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir

    def log_metrics(self, metrics, step):
        epoch = metrics.pop("epoch")

        if "train_loss" in metrics:
            filename = os.path.join(self.log_dir, "train.json")
        elif "val_loss" in metrics:
            filename = os.path.join(self.log_dir, "val.json")
            epoch += 1
        else:
            filename = os.path.join(self.log_dir, "metrix.json")

        #train.json, val.json, test.jsonがなければ作成
        if os.path.isfile(filename) == False:
            with open(filename, "w+") as f:
                json.dump({}, f)
        
        with open(filename, 'r') as f:
            log = json.load(f)
        with open(filename, 'w') as f:
            log[epoch] = metrics
            json.dump(log, f)

    @property
    def experiment(self):
        pass

    @property
    def name(self):
        return 'JsonLogger'

    def log_hyperparams(self, hparams):
        pass

    @property
    def version(self):
        pass

class DatasetFromPath(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], self.img_paths[idx]


def make_K_dataset(train_dir, train_transform, valid_transform, return_one_fold=True, n_split=4):
    class_names = sorted(os.listdir(train_dir))
    # ファイルパスとラベル名を格納 (あとで複数インデックスで一気に取得したいのでnumpyに変換しとく)
    img_paths = np.array(sorted(glob.glob(os.path.join(train_dir, "*", "*"))))
    labels = np.array([], dtype="int")
    for path in img_paths:
        dir_path = os.path.dirname(path)
        class_name = os.path.basename(dir_path)
        label = class_names.index(class_name)
        labels = np.append(labels, label)

    train_datasets = []
    valid_datasets = []
    kf = StratifiedKFold(n_split, shuffle=True, random_state=1)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(img_paths, labels)):
        train_dataset = DatasetFromPath(img_paths[train_idx], labels[train_idx], train_transform)
        valid_dataset = DatasetFromPath(img_paths[valid_idx], labels[valid_idx], valid_transform)
        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)
        # 単一のtrain_dataset, valid_datasetの組を返す場合
        if return_one_fold:
            break
    return train_datasets, valid_datasets, class_names
