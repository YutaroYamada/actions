import os
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import build_model, LitModel, JsonLogger, make_K_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--valid_rate', type=float, default=0.25)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--model_name', type=str, default='resnet18')
parser.add_argument('--pretrained', type=bool, default=False)
args = parser.parse_args()

# sagemakerで動かす都合この固定パス
TRAINING_DIR = '/opt/ml/input/data/training'
OUTPUT_DIR_PATH = '/opt/ml/checkpoints'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

train_transform = T.Compose([
    T.Resize((args.image_size)), #横幅はimage_sizeにリサイズされる。縦幅は, 元画像の縦横比を保って, 横幅と同じ縮小率でリサイズされる。
    T.ToTensor(),
    T.ColorJitter(brightness=(0.7, 1.3), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=0),
    T.RandomHorizontalFlip(p=0.3),
    T.RandomVerticalFlip(p=0.3),
    T.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1), scale=(0.7, 1.2), fill=(0.0)), 
#     T.RandomGrayscale(p=0.3),
])

valid_transform = T.Compose([
    T.Resize((args.image_size)),
    T.ToTensor(),
])

n_split = int(1/args.valid_rate) #これ何とかしたい
train_datasets, valid_datasets, class_names = make_K_dataset(TRAINING_DIR, train_transform, valid_transform, return_one_fold=True, n_split=n_split)

train_dataloader = DataLoader(train_datasets[0], batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, drop_last=True, persistent_workers=False)
valid_dataloader = DataLoader(valid_datasets[0], batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=False, persistent_workers=False)
print(f'batchsize:{args.batch_size}, train_dataset_size:{len(train_datasets[0])}, valid_dataset_size:{len(valid_datasets[0])}')


model = build_model(model_name=args.model_name, class_num=len(class_names), pretrained=args.pretrained)
lit_model = LitModel(model, class_names)

logger = JsonLogger(log_dir=OUTPUT_DIR_PATH)
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=OUTPUT_DIR_PATH, filename="best")

# Initialize a trainer
trainer = pl.Trainer(
    max_epochs=args.epochs,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    precision=16 if torch.cuda.is_available() else 32,
    logger=logger,
    callbacks=[early_stop_callback, checkpoint_callback],
    num_sanity_val_steps=0
)

# モデルを学習
trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)