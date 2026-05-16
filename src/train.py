#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License: Opensource, free to use
Other: Suggestions are welcome
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from manage.CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from manage.HDF5Dataset import HDF5Dataset
from models.AlexNet import AlexNet
from models.CNNVanilla import CnnVanilla
from models.ResNet import ResNet
from models.UNet import UNet
from models.yourSegNet import YourSegNet
from models.VggNet import VggNet
from models.yourUNet import YourUNet
from losses import DiceLoss
from torchvision import datasets


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs, dims) + torch.sum(targets_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def argument_parser():
    """
    A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(
        usage="\n python3 train.py [model] [dataset] [hyper_parameters]"
        "\n python3 train.py --model UNet [hyper_parameters]"
        "\n python3 train.py --model UNet --predict",
        description="This program allows to train different models of classification on"
        " different datasets. Be aware that when using UNet model there is no"
        " need to provide a dataset since UNet model only train "
        "on acdc dataset.",
        add_help=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CnnVanilla",
        choices=[
            "CnnVanilla",
            "VggNet",
            "AlexNet",
            "ResNet",
            "yourUNet",
            "yourSegNet",
            "UNet",
        ],
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "svhn"]
    )
    parser.add_argument("--loss", type=str, default="CE")
    parser.add_argument(
        "--batch_size", type=int, default=20, help="The size of the training batch"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "SGD"],
        help="The optimizer to use for training the model",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=10, help="The number of epochs"
    )
    parser.add_argument(
        "--validation",
        type=float,
        default=0.1,
        help="Percentage of training data to use for validation",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_aug", action="store_true", help="Data augmentation")
    parser.add_argument(
        "--predict",
        type=str,
        help="Name of the file containing model weights used to make "
        "segmentation prediction on test data",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr
    data_augment = args.data_aug
    if data_augment:
        print("Data augmentation activated!")
    else:
        print("Data augmentation NOT activated!")

    # set hdf5 path according your hdf5 file location
    acdc_hdf5_file = 'data/ift780_acdc.hdf5'

    # Transform is used to normalize data among others
    acdc_base_transform = transforms.Compose([transforms.ToTensor()])

    acdc_augment_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ]
    )

    data_augment_transform = transforms.Compose(
        [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    )
    base_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    transform = base_transform
    if data_augment:
        transform = transforms.Compose([data_augment_transform, transform])

    if args.dataset == "cifar10":
        # Download the train and test set and apply transform on it
        train_set = datasets.CIFAR10(
            root="../data", train=True, download=True, transform=transform
        )
        test_set = datasets.CIFAR10(
            root="../data", train=False, download=True, transform=base_transform
        )

    elif args.dataset == "svhn":
        # Download the train and test set and apply transform on it
        train_set = datasets.SVHN(
            root="../data", split="train", download=True, transform=transform
        )
        test_set = datasets.SVHN(
            root="../data", split="test", download=True, transform=base_transform
        )

    if args.optimizer == "SGD":
        optimizer_factory = optimizer_setup(
            torch.optim.SGD, lr=learning_rate, momentum=0.9
        )
    elif args.optimizer == "Adam":
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.model == "CnnVanilla":
        model = CnnVanilla(num_classes=10)
    elif args.model == "AlexNet":
        model = AlexNet(num_classes=10)
    elif args.model == "VggNet":
        model = VggNet(num_classes=10)
    elif args.model == "ResNet":
        model = ResNet(num_classes=10)
    elif args.model == 'yourSegNet':
        model = YourSegNet(num_classes=4)
        args.dataset = 'acdc'
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform, augment=data_augment)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)
    elif args.model == 'yourUNet':
        model = YourUNet(num_classes=4)
        args.dataset = 'acdc'
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform, augment=data_augment)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)
    elif args.model == 'UNet':
        model = UNet(num_classes=4)
        args.dataset = 'acdc'
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=acdc_base_transform, augment=data_augment)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=acdc_base_transform)

    if args.model in ['yourSegNet', 'UNet', 'yourUNet']:
        if args.loss.lower() in ['ce', 'crossentropy']:
            loss_fn = nn.CrossEntropyLoss()
        elif args.loss.lower() in ['dice']:
            loss_fn = DiceLoss()
        else:
            loss_fn = CombinedSegmentationLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    model_trainer = CNNTrainTestManager(model=model,
                                        trainset=train_set,
                                        testset=test_set,
                                        batch_size=batch_size,
                                        loss_fn=loss_fn,
                                        optimizer_factory=optimizer_factory,
                                        validation=val_set,
                                        use_cuda=True,
                                        checkpoint_path=f'best_{args.model}.pt')

    if args.predict is not None:
        model.load_weights(args.predict)
        print("predicting the mask of a randomly selected image from test set")
        model_trainer.plot_image_mask_prediction()
    else:
        print(
            "Training {} on {} for {} epochs".format(
                model.__class__.__name__, args.dataset, args.num_epochs
            )
        )
        model_trainer.train(num_epochs)
        model_trainer.evaluate_on_test_set()
        if isinstance(model, UNet):
            model.save()  # save the model's weights for prediction (see help for more details)
            model_trainer.plot_image_mask_prediction()
        model_trainer.plot_metrics()
    