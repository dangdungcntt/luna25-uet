#train.py
"""
Script for training a Pulse-3D to classify a pulmonary nodule as benign or malignant.
"""
from models.model_2d import ResNet18
from models.model_3d import I3D
from models.torchvision_video import TorchvisionModels
from models.pulse_3d import Pulse3D
from models.mclab import I3D_UNet_MCLab
from models.unet_3d import I3D_UNet
from dataloader import get_data_loader
from loss_function import *
from experiment_config import config

import wandb    
import random
import logging
from datetime import datetime
from tqdm import tqdm
import sklearn.metrics as metrics
import pandas
import numpy as np
import torch


torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

def make_weights_for_balanced_classes(labels):
    """Making sampling weights for the data samples
    :returns: sampling weights for dealing with class imbalance problem

    """
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        weights.append(n_samples / float(cnt_dict[label]))
    return weights

def iou_3d(y_true, y_pred):
    """
    y_true: numpy array of shape (B, 1, D, H, W), binary {0,1}
    y_pred: numpy array of shape (B, 1, D, H, W), probabilities or logits
    """
    # Binarize predictions
    y_pred_bin = y_pred.astype(np.uint8)
    y_true_bin = y_true.astype(np.uint8)

    batch_size = y_true_bin.shape[0]
    ious = []

    for i in range(batch_size):
        # Remove channel dimension and flatten
        y_t = y_true_bin[i, 0].reshape(-1)
        y_p = y_pred_bin[i, 0].reshape(-1)

        intersection = np.logical_and(y_t, y_p).sum()
        union = np.logical_or(y_t, y_p).sum()

        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)

    return np.mean(ious)  # mean IoU

def train(
    train_csv_path,
    valid_csv_path,
    exp_save_root,
):
    """
    Train a model
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    for key, value in vars(config).items():
        logging.info(f"{key} : {value}")

    train_df = pandas.read_csv(train_csv_path)
    valid_df = pandas.read_csv(valid_csv_path)

    print()

    logging.info(
        f"Number of malignant training samples: {train_df.label.sum()}"
    )
    logging.info(
        f"Number of benign training samples: {len(train_df) - train_df.label.sum()}"
    )
    print()
    logging.info(
        f"Number of malignant validation samples: {valid_df.label.sum()}"
    )
    logging.info(
        f"Number of benign validation samples: {len(valid_df) - valid_df.label.sum()}"
    )

    # create a training data loader
    weights = make_weights_for_balanced_classes(train_df.label.values)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))

    train_loader = get_data_loader(
        config.DATADIR,
        config.MASKDIR,
        train_df,
        mode=config.MODE,
        sampler=sampler,
        balanced=config.BALANCED,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        flip_probs=config.FLIP_PROBS,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
        interpolate_scale=config.INTERPOLATE_SCALE,
        use_monai_transforms=config.USE_MONAI_TRANSFORMS
    )

    valid_loader = get_data_loader(
        config.DATADIR,
        config.MASKDIR,
        valid_df,
        mode=config.MODE,
        sampler=None,
        balanced=False,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        flip_probs=[0] * 3 if config.MODE == "3D" else [0] * 2,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
        interpolate_scale=config.INTERPOLATE_SCALE,
        use_monai_transforms=False
    )

    device = torch.device(config.DEVICE)

    if config.MODE == "2D":
        model = ResNet18().to(device)
    elif config.MODE == "3D":
        # model = I3D(
        #     num_classes=1,
        #     input_channels=3,
        #     pre_trained=True,
        #     freeze_bn=True,
        # ).to(device)

        # model = TorchvisionModels(
        #     model_name=config.MODEL_NAME,
        #     num_classes=1,
        #     pre_trained=True,
        # ).to(device)

        # model = Pulse3D(
        #     num_classes=1,
        #     input_channels=3,
        #     pre_trained=True,
        #     freeze_bn=False,
        # ).to(device)

        # model = I3D_UNet_MCLab(
        #     num_classes=1,
        #     input_channels=3,
        #     out_channels=1,
        #     pre_trained=True,
        #     freeze_bn=True,
        # ).to(device)

        model = I3D_UNet(
            num_classes=1,
            input_channels=3,
            out_channels=1,
            pre_trained=True,
            freeze_bn=True,
        ).to(device)

    start_epoch = 0
    resume_checkpoint = config.CKPT_PATH
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        print("Resuming at epoch", start_epoch)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"Resumed training")

    loss_function = torch.nn.BCEWithLogitsLoss()
    # loss_function = FocalLoss()
    # mask_function = torch.nn.BCEWithLogitsLoss()
    mask_function = DiceLoss()
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=config.LEARNING_RATE,
    #     weight_decay=config.WEIGHT_DECAY,
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer,
    #     mode='max',
    #     factor=0.1,
    #     patience=5,
    # )
    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epochs = config.EPOCHS

    for epoch in range(start_epoch, epochs):
        logging.info("-" * 10)
        logging.info("epoch {}/{}".format(epoch + 1, epochs))

        # train

        model.train()

        epoch_loss = {
            "loss": 0,
            "class_loss": 0,
            "segment_loss": 0
        }
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["image"], batch_data["label"]
            masks = batch_data['mask']

            labels = labels.float().to(device)
            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            cls_outputs, seg_outputs = model(inputs)

            cls_loss = loss_function(cls_outputs.squeeze(), labels.squeeze())
            seg_loss = config.AUX_LOSS_WEIGHT * mask_function(seg_outputs, masks)
            loss = cls_loss + seg_loss
            loss.backward()
            optimizer.step()

            # epoch_loss += loss.item()
            epoch_loss['loss'] += loss.item()
            epoch_loss['class_loss'] += cls_loss.item()
            epoch_loss['segment_loss'] += seg_loss.item()

            epoch_len = len(train_df) // train_loader.batch_size
            if step % 100 == 0:
                logging.info(
                    "{}/{}, train_loss: {:.7f}".format(step, epoch_len, loss.item())
                )
        # epoch_loss /= step

        log_data = {f"train/{k}" : (v / step) for k, v in epoch_loss.items()}
        log_data['train/lr'] = optimizer.param_groups[0]['lr']
        wandb.log(log_data)

        logging.info(
            "epoch {} average train loss: {:.7f}".format(epoch + 1, log_data['train/loss'])
        )

        # validate

        model.eval()

        epoch_loss = {
            "loss": 0,
            "class_loss": 0,
            "segment_loss": 0
        }
        step = 0

        with torch.no_grad():

            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y_seg_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.float32, device=device)
            y_seg = torch.tensor([], dtype=torch.float32, device=device)
            for val_data in valid_loader:
                step += 1
                val_images, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_masks = val_data['mask'].to(device)

                val_images = val_images.to(device)
                val_labels = val_labels.float().to(device)
                cls_outputs, seg_outputs = model(val_images)
                cls_loss = loss_function(cls_outputs.squeeze(), val_labels.squeeze())
                mask_loss = config.AUX_LOSS_WEIGHT * mask_function(seg_outputs, val_masks)
                loss = cls_loss + mask_loss

                # epoch_loss += loss.item()
                epoch_loss['loss'] += loss.item()
                epoch_loss['class_loss'] += cls_loss.item()
                epoch_loss['segment_loss'] += seg_loss.item()

                y_pred = torch.cat([y_pred, cls_outputs], dim=0)
                y_seg_pred = torch.cat([y_seg_pred, seg_outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                y_seg = torch.cat([y_seg, val_masks], dim=0)

                epoch_len = len(valid_df) // valid_loader.batch_size

            log_data = {f"val/{k}" : (v / step) for k, v in epoch_loss.items()}
            
            logging.info(
                "epoch {} average valid loss: {:.7f}".format(epoch + 1, log_data['val/loss'])
            )

            y_pred = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
            y = y.data.cpu().numpy().reshape(-1)
            
            y_seg_pred = torch.sigmoid(y_seg_pred).data.cpu().numpy() >= config.THRESHOLD
            y_seg = y_seg.data.cpu().numpy()

            fpr, tpr, _ = metrics.roc_curve(y, y_pred)
            auc_metric = metrics.auc(fpr, tpr)

            y_pred_thresh = y_pred >= config.THRESHOLD
            acc = metrics.accuracy_score(y, y_pred_thresh)
            f1 = metrics.f1_score(y, y_pred_thresh)
            recall = metrics.recall_score(y, y_pred_thresh)
            precision = metrics.precision_score(y, y_pred_thresh)
            iou = iou_3d(y_seg, y_seg_pred)

            # scheduler.step(auc_metric)
            log_data.update({
                "val/auroc": auc_metric,
                "val/acc": acc,
                "val/precision": precision,
                "val/recall": recall,
                "val/f1": f1,
                "val/iou": iou,
                "epoch": epoch
            })
            wandb.log(log_data)
            
            torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch
                    },
                    exp_save_root / "last.pth",
                )

            if auc_metric > best_metric:

                counter = 0
                best_metric = auc_metric
                best_metric_epoch = epoch + 1

                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch
                    },
                    exp_save_root / f"best_ckpt.pth",
                )

                metadata = {
                    "train_csv": train_csv_path,
                    "valid_csv": valid_csv_path,
                    "config": config,
                    "best_auc": best_metric,
                    "epoch": best_metric_epoch,
                }
                np.save(
                    exp_save_root / "config.npy",
                    metadata,
                )

                logging.info("saved new best metric model")

            logging.info(
                "current epoch: {} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                    epoch + 1, auc_metric, best_metric, best_metric_epoch
                )
            )
        counter += 1

    logging.info(
        "train completed, best_metric: {:.4f} at epoch: {}".format(
            best_metric, best_metric_epoch
        )
    )

def test(
    test_csv_path,
    exp_save_root,    
):
    test_df = pandas.read_csv(test_csv_path)
    test_loader = get_data_loader(
        config.DATADIR,
        config.MASKDIR,
        test_df,
        mode=config.MODE,
        sampler=None,
        balanced=None,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        flip_probs=[0] * 3 if config.MODE == "3D" else [0] * 2,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
        interpolate_scale=config.INTERPOLATE_SCALE,
        use_monai_transforms=False
    )

    device = torch.device(config.DEVICE)

    if config.MODE == "2D":
        model = ResNet18().to(device)
    elif config.MODE == "3D":
        # model = I3D(
        #     num_classes=1,
        #     input_channels=3,
        #     pre_trained=True,
        #     freeze_bn=True,
        # ).to(device)

        # model = TorchvisionModels(
        #     model_name=config.MODEL_NAME,
        #     num_classes=1,
        #     pre_trained=True,
        # ).to(device)

        # model = Pulse3D(
        #     num_classes=1,
        #     input_channels=3,
        #     pre_trained=True,
        #     freeze_bn=False,
        # ).to(device)

        # model = I3D_UNet_MCLab(
        #     num_classes=1,
        #     input_channels=3,
        #     out_channels=1,
        #     pre_trained=True,
        #     freeze_bn=True,
        # ).to(device)

        model = I3D_UNet(
            num_classes=1,
            input_channels=3,
            out_channels=1,
            pre_trained=True,
            freeze_bn=True,
        ).to(device)

    # Testing
    for i, ckpt_file in enumerate(["best_ckpt.pth", "last.pth"]):
        print(f"\n========== Testing {ckpt_file}==========")
        best_state_dict = torch.load(exp_save_root / ckpt_file, map_location=device)
        print(f"Loading {'best' if i == 0 else 'last'} checkpoint at epoch", best_state_dict["epoch"] + 1)
        model.load_state_dict(best_state_dict['state_dict'])

        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.float32, device=device)
            y_seg_pred = torch.tensor([], dtype=torch.float32, device=device)
            y_seg = torch.tensor([], dtype=torch.float32, device=device)
            for test_data in tqdm(test_loader):
                test_images, test_labels, test_masks = (
                    test_data["image"].to(device),
                    test_data["label"].to(device),
                    test_data['mask'].to(device)
                )
                test_images = test_images.to(device)
                test_labels = test_labels.float().to(device)
                cls_outputs, seg_outputs = model(test_images)

                y_pred = torch.cat([y_pred, cls_outputs], dim=0)
                y = torch.cat([y, test_labels], dim=0)

                y_seg_pred = torch.cat([y_seg_pred, seg_outputs], dim=0)
                y_seg = torch.cat([y_seg, test_masks], dim=0)

            y_pred = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
            y = y.data.cpu().numpy().reshape(-1)

            y_seg_pred = torch.sigmoid(y_seg_pred).data.cpu().numpy() >= config.THRESHOLD
            y_seg = y_seg.data.cpu().numpy()

            fpr, tpr, _ = metrics.roc_curve(y, y_pred)
            auc_metric = metrics.auc(fpr, tpr)

            y_pred_thresh = y_pred >= config.THRESHOLD
            acc = metrics.accuracy_score(y, y_pred_thresh)
            f1 = metrics.f1_score(y, y_pred_thresh)
            recall = metrics.recall_score(y, y_pred_thresh)
            precision = metrics.precision_score(y, y_pred_thresh)
            iou = iou_3d(y_seg, y_seg_pred)

            wandb.log(
                {
                "test/auroc": auc_metric,
                "test/acc": acc,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1,
                "test/iou": iou,
                }
            )

if __name__ == "__main__":
    experiment_name = config.RUN_NAME
    exp_save_root = config.EXPERIMENT_DIR / "classify" / experiment_name
    exp_save_root.mkdir(parents=True, exist_ok=True)

    wandb.init(
            project="Luna25",
            name=experiment_name,
            id=config.WANDB_ID,
            tags=[],
            resume="auto" if config.CKPT_PATH is not None else None,
            config=vars(config),
        )
    # start training run
    train(
        train_csv_path=config.CSV_DIR_TRAIN,
        valid_csv_path=config.CSV_DIR_VALID,
        exp_save_root=exp_save_root,
    )

    test(
        test_csv_path=config.CSV_DIR_VALID,
        exp_save_root=exp_save_root,
    )