# ResNet18 ArcFace Trainer

Standalone ResNet18-only training code extracted from `arcface-pytorch`.

The default dataset path is:

```text
dataset/11-785-fall-20-homework-2-part-2/classification_data
```

Expected layout:

```text
classification_data/
  train_data/<class_name>/*.jpg
  val_data/<class_name>/*.jpg
  test_data/<class_name>/*.jpg
```

Run:

```bash
python train_resnet18.py --epochs 50 --batch-size 128 --num-workers 4 --amp
```

Training plots are generated after training finishes:

```text
checkpoints_resnet18/training_curves.png
checkpoints_resnet18/validation_metrics.png
```

Useful options:

```bash
python train_resnet18.py --head arcface --loss focal
python train_resnet18.py --head linear --loss ce
python train_resnet18.py --optimizer adamw --scheduler cosine --lr 0.001 --min-lr 0.000001
python train_resnet18.py --freeze-backbone 3
python train_resnet18.py --image-size 128 --pretrained-backbone weights/resnet18_110.pth
python train_resnet18.py --image-size 128 --resize-size 144
python train_resnet18.py --resume checkpoints_resnet18/last.pth
```

Preprocessing follows the original ArcFace loader: grayscale, train random crop, train horizontal flip, eval center crop, tensor conversion, then normalization with mean/std `0.5`. `--resize-size` runs before the crop; keep it larger than `--image-size` for random crop behavior, or set `--resize-size 0` to disable resizing.

`--freeze-backbone` freezes stages from the input side forward:

```text
0 = no freeze
1 = stem + layer1
2 = stem + layer1 + layer2
3 = stem + layer1 + layer2 + layer3
4 = add layer4
5 = add embedding tail
```

Checkpoints, `training_metadata.json`, and `metrics.csv` are written to `checkpoints_resnet18/` by default. Test confusion matrix outputs are saved as:

```text
checkpoints_resnet18/training_history.json
checkpoints_resnet18/test_confusion_matrix.pt
checkpoints_resnet18/test_confusion_matrix_nonzero.csv
```
