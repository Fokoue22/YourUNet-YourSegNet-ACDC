# YourUNet-YourSegNet-ACDC

Advanced cardiac MRI segmentation using state-of-the-art deep learning architectures. This project implements multiple UNet-based models for segmenting cardiac structures (right ventricle, myocardium, left ventricle) from 2D cardiac cine-MRI images using the ACDC dataset.

##  Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Architecture](#architecture)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Data Augmentation](#data-augmentation)
- [Checkpointing](#checkpointing)

## Features

###  Implemented Architectures
- **UNet**: Classic encoder-decoder architecture for medical image segmentation
- **YourUNet**: Enhanced UNet with:
  - Residual blocks (ResNet-style skip connections)
  - Depthwise separable convolutions for efficiency
  - ASPP (Atrous Spatial Pyramid Pooling) module at bottleneck for multi-scale context
  
- **YourSegNet**: Non-symmetric encoder-decoder with:
  - Aggressive downsampling encoder (1/2 → 1/4 → 1/8 → 1/16)
  - Lightweight decoder with dilated convolutions
  - Multi-scale context aggregation via ASPP
  - Detail fusion for enhanced segmentation

###  Advanced Training Features
- **Checkpointing**: Automatic saving of best model during training based on validation accuracy
- **Custom Loss Functions**:
  - Dice Loss: Specifically designed for medical image segmentation
  - Combined Loss: Weighted combination of Cross-Entropy and Dice Loss
  - Cross-Entropy Loss: Baseline loss function
- **Data Augmentation**:
  - Random horizontal flips
  - Random rotations (±15°)
  - RandomAffine transformations
- **Learning Rate Scheduling**: ReduceLROnPlateau scheduler for adaptive learning rate
- **Model Zoo**: Includes additional architectures (ResNet, AlexNet, VGG) for classification tasks

##  Installation

### Prerequisites
- Python 3.7+
- CUDA 11.0+ (for GPU acceleration)
- PyTorch 1.9+

### Setup
Install dependencies in a Python virtual environment:

```bash
pip install -r requirements.txt
```

Download the ACDC dataset (HDF5 format) and place it in the `data/` directory:
```
data/ift780_acdc.hdf5
```

## Architecture

### Model Architectures

#### YourUNet Architecture
![YourUNet Architecture](diagrams/yourUNet_architecture.png)

**Modified Files:**
- `src/models/yourUNet.py` - Complete YourUNet implementation with DepthwiseSeparableConv, ResidualBlock, ASPPModule, and YourUNet classes
- `src/train.py` - Added yourUNet instantiation and ACDC dataset loading

**Architecture Components:**
- **DepthwiseSeparableConv**: Depthwise convolution followed by 1×1 pointwise convolution for efficient feature extraction
- **ResidualBlock**: Two convolutions (standard or separable) with learnable shortcuts (shortcut) for improved gradient flow
- **ASPPModule**: 4 parallel branches with 1×1 conv, dilations (3, 6), and global average pooling concatenated and fused

**YourUNet Structure:**
- **Encoder**: 4-level pyramid with ResidualBlocks (1→64→128→256→512 channels)
- **Bottleneck**: ASPP module at lowest resolution for multi-scale context
- **Decoder**: 4-level symmetric upsampling with skip connections and transposed convolutions
- **Output**: Final 1×1 convolution mapping to 4 classes

#### YourSegNet Architecture
![YourSegNet Architecture](diagrams/yourSegNet_architecture.png)

**Modified Files:**
- `src/models/yourSegNet.py` - Complete YourSegNet implementation with ConvBNReLU, ResidualBlock, ASPP, and ASPPBlock classes
- `src/train.py` - Added yourSegNet instantiation and ACDC dataset loading

**Architecture Components:**
- **ConvBNReLU**: Convolution + BatchNorm + ReLU block with support for dilated convolutions
- **ResidualBlock**: Conv with optional stride and learnable shortcut for identity mapping
- **ASPP**: Multi-scale context module with 4 branches (1×1, dilations 6/12/18, global pooling)
- **ASPPBlock**: Decoder block with 2× bilinear upsampling and dilated convolutions

**YourSegNet Structure (Non-symmetric):**
- **Encoder** (aggressive downsampling): Stem (1/2) → Enc1 (1/4) → Enc2 (1/8) → Enc3 (1/16)
- **Bottleneck**: ASPP module for multi-scale context aggregation
- **Decoder** (lightweight): Dec1 (1/8) → Dec2 (1/4) → Dec3 (1/2) with upsampling
- **Detail Fusion**: 1×1 convolution on stem features fused via element-wise addition
- **Output**: Final 1×1 convolution to 4 classes with bilinear interpolation to input resolution

---

### Loss Functions

**Modified Files:**
- `src/losses.py` - New file containing DiceLoss class
- `src/train.py` - Added DiceLoss and CombinedSegmentationLoss classes with conditional selection

#### Dice Loss
Specifically designed for medical image segmentation to maximize overlap between prediction and ground truth.

**Formula:**
$$\text{DiceLoss} = 1 - \frac{1}{C}\sum_{c=1}^{C}\frac{2|P_c \cap Y_c| + \varepsilon}{|P_c| + |Y_c| + \varepsilon}$$

Where:
- $C$ = number of classes
- $P_c$ = predicted probability for class $c$
- $Y_c$ = ground truth for class $c$  
- $\varepsilon$ = smoothing factor (default: 1e-6)

**Advantage:** Critical for ACDC dataset where small structures (myocardium) have imbalanced class distributions.

#### Combined Loss
Weighted combination of Cross-Entropy and Dice Loss for robust training.

**Formula:**
$$L_{\text{Combined}} = 0.5 \times L_{\text{CE}} + 0.5 \times L_{\text{Dice}}$$

**Benefit:** Combines numerical stability of Cross-Entropy with inter-class sensitivity of Dice, particularly effective for imbalanced segmentation tasks.

#### Cross-Entropy Loss
Baseline loss function for multi-class classification (used with `--loss CE`).


## Usage

View all available options:
```bash
python src/train.py --help
```

### Training Models

**Train UNet (baseline):**
```bash
python src/train.py --model UNet --num-epochs 40 --batch_size 20 --lr 0.001
```

**Train YourUNet with data augmentation:**
```bash
python src/train.py --model yourUNet --num-epochs 40 --batch_size 20 --lr 0.001 --data_aug
```

**Train YourSegNet with Dice Loss:**
```bash
python src/train.py --model yourSegNet --num-epochs 40 --batch_size 20 --lr 0.001 --loss dice --data_aug
```

**Train with combined loss function:**
```bash
python src/train.py --model yourUNet --num-epochs 40 --batch_size 20 --lr 0.001 --loss combined
```

### Available Options
```
--model {CnnVanilla, VggNet, AlexNet, ResNet, yourUNet, yourSegNet, UNet}
--dataset {cifar10, svhn, acdc}
--loss {CE, Dice, Combined}
--batch_size (int, default=20)
--optimizer {Adam, SGD}
--num-epochs (int, default=10)
--validation (float, default=0.1)
--lr (float, default=0.001)
--data_aug (flag to enable data augmentation)
```

## Training

### Learning Curves and Results

Training logs are automatically saved and visualized. The trainer:
1. Displays real-time training/validation metrics with progress bars
2. Saves the best model checkpoint (e.g., `best_yourUNet.pt`)
3. Generates learning curves and performance plots at the end of training

### Measured Performance

**YourUNet Results** (15 epochs, batch_size=8, lr=0.001, with data augmentation):
- **Validation Accuracy**: ~85% after 15 epochs
- **Training Convergence**: Extremely rapid, stable loss by epoch 10
- **Generalization**: Strong, with reduced gap between training and validation metrics
- **Training Loss**: Drops from ~0.53 to ~0.10 in first 10 epochs
- **Key Observation**: Depthwise separable convolutions + ASPP module significantly improve convergence speed

**YourSegNet Results** (40 epochs, batch_size=8, lr=0.0005, with combined loss):
- **Validation Accuracy**: ~80% after 40 epochs
- **Training Convergence**: Rapid initial convergence with steady improvement
- **Generalization**: Good, validation accuracy follows training without significant divergence
- **Scheduler Impact**: ReduceLROnPlateau adjustments visible as oscillations between epochs 15-25
- **Key Observation**: Non-symmetric design achieves required >75% accuracy with lighter decoder

**UNet (Baseline)** (40 epochs, standard settings):
- **Validation Accuracy**: ~79% validation accuracy
- **Performance**: Solid baseline, outperformed by custom architectures

### Performance Analysis

- **YourUNet** demonstrates superior convergence due to residual blocks improving gradient flow and depthwise separable convolutions reducing computational burden
- **YourSegNet** efficiently achieves >75% accuracy with asymmetric design, suitable for resource-constrained environments
- **Combined Loss Function** provides stability by balancing cross-entropy's numerical properties with Dice's class-balance sensitivity

## Data Augmentation

**Modified Files:**
- `src/train.py` - Added `acdc_augment_transform` with conditional selection and `augment` parameter
- `src/manage/HDF5Dataset.py` - Added `RandomSegmentationAugmentation` class for synchronized image-mask augmentation

Data augmentation is applied to improve model generalization and robustness:

```bash
python src/train.py --model yourUNet --data_aug
python src/train.py --model yourSegNet --data_aug
```

### Augmentation Types

**For YourUNet (PyTorch transforms in train.py):**
- **Random Horizontal Flip** (p=0.5): Flips images and masks left-right using `transforms.RandomHorizontalFlip`
- **Random Rotation** (±15°): Rotates images and masks by -15° to +15° using `transforms.RandomRotation`

**For YourSegNet (Synchronized augmentation in HDF5Dataset.py):**
- **Random Horizontal Flip** (p=0.5): Applied via `np.fliplr()` simultaneously to image and mask
- **Random 90° Rotation**: Applies multiples of 90° rotations (k ∈ {0,1,2,3}) via `np.rot90()` to both image and mask simultaneously

**Augmentation Strategy:**
Geometric transformations only (no color jittering) are used for single-channel cardiac MRI:
- Preserves medical image integrity
- Maintains pixel value ranges critical for MRI analysis
- Synchronized transformations ensure mask correctness

### Augmentation Examples
See `fig_augmentation_examples.png` for visual examples showing the effect of augmentations on cardiac MRI images and corresponding segmentation masks.

## Checkpointing

**Modified Files:**
- `src/manage/CNNTrainTestManager.py` - Added `checkpoint_path` parameter, best validation accuracy tracking, and conditional model saving
- `src/train.py` - Passes `checkpoint_path=f'best_{args.model}.pt'` to trainer

### How Checkpointing Works

The training manager implements automatic model checkpointing during training:

1. **Tracking**: Maintains `self.best_val_acc` to track the highest validation accuracy achieved
2. **Saving**: When current validation accuracy exceeds `best_val_acc`, model weights are saved via `torch.save(model.state_dict(), checkpoint_path)`
3. **Scheduler**: `ReduceLROnPlateau` scheduler reduces learning rate by factor of 0.5 when validation loss plateaus (patience=3 epochs)

### Checkpoint Files

Best models are automatically saved with descriptive names:
```
best_yourUNet.pt      # Best YourUNet checkpoint
best_yourSegNet.pt    # Best YourSegNet checkpoint  
best_UNet.pt          # Best baseline UNet checkpoint
best_model.pt         # Additional backup (from model.save())
```

### Training Commands with Checkpointing

```bash
# Checkpoints automatically saved to best_yourUNet.pt
python src/train.py --model yourUNet --num-epochs 15 --batch_size 8 --lr 0.001 --data_aug

# Checkpoints automatically saved to best_yourSegNet.pt
python src/train.py --model yourSegNet --num-epochs 40 --batch_size 8 --lr 0.0005 --loss combined
```

### Benefits
- **Fault Tolerance**: Best model preserved even if training is interrupted by power failure
- **Optimal Model Recovery**: Always have access to the model with best validation performance
- **Training Monitoring**: View checkpoint saves in console output: "Saved best checkpoint to best_<model>.pt (val_acc=0.XXX)"

## Project Structure

```
YourUNet-YourSegNet-ACDC/
├── src/
│   ├── train.py                 # Main training script
│   ├── losses.py                # Custom loss functions
│   ├── train.ipynb              # Jupyter notebook for experimentation
│   ├── manage/
│   │   ├── CNNTrainTestManager.py  # Training & testing manager
│   │   ├── DataManager.py          # Data loading utilities
│   │   └── HDF5Dataset.py          # HDF5 dataset handler
│   ├── models/
│   │   ├── UNet.py              # Standard UNet
│   │   ├── yourUNet.py          # Enhanced UNet with residuals + ASPP
│   │   ├── yourSegNet.py        # Non-symmetric architecture
│   │   ├── CNNBaseModel.py      # Base model class
│   │   ├── CNNBlocks.py         # Reusable blocks
│   │   ├── ResNet.py            # ResNet classifier
│   │   ├── AlexNet.py           # AlexNet classifier
│   │   └── VggNet.py            # VGG classifier
│   └── utils/
│       └── utils.py             # Utility functions
├── data/
│   └── ift780_acdc.hdf5         # ACDC dataset (download required)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## Requirements

See `requirements.txt` for complete list. Key dependencies:
- torch & torchvision
- h5py (HDF5 dataset handling)
- numpy, scipy, scikit-image
- matplotlib (visualization)
- tqdm (progress bars)

## References

- ACDC Dataset: https://www.creatis.insa-lyon.fr/Challenge/acdc/
- Original UNet: https://arxiv.org/abs/1505.04597
- DeepLabV3: https://arxiv.org/abs/1706.05587
- ASPP: https://arxiv.org/abs/1706.05587

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- **Fokoue Thomas** - Custom architectures (YourUNet, YourSegNet), loss functions, data augmentation
- **Mamadou Mountagha BAH & Pierre-Marc Jodoin** - Original UNet framework and training infrastructure

Modified and extended for advanced cardiac MRI segmentation with research-grade implementations.
