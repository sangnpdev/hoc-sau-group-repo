# FER2013 Emotion Recognition - Experiment Report

## Setup

- Framework: PyTorch; runtime: Google Colab Tesla T4 GPU.
- Input: 48x48 grayscale facial images; seven classes in the assignment order: angry, disgust, fear, happy, sad, surprise, neutral.
- Split: 28,709 training images, 3,589 PublicTest validation images, and 3,589 PrivateTest images reserved for the final evaluation.
- All models use two 3x3 convolution layers per block, batch normalization, ReLU, 2x2 max pooling, dropout, adaptive average pooling, and a 256-unit fully connected layer.
- Each run was capped at 12 epochs. The best checkpoint for each model was selected by validation loss.

## Experiments

| Experiment | Blocks (channels) | Augmentation | Optimizer settings | Best epoch | Validation accuracy | Validation loss |
|---|---|---|---|---:|---:|---:|
| small | 3: 32, 64, 128 | None | AdamW, lr=0.001, weight decay=0, dropout=0.25 | 12 | 60.04% | 1.039 |
| medium_aug | 4: 32, 64, 128, 256 | Horizontal flip, rotation, translation, scale | AdamW, lr=0.001, weight decay=0, dropout=0.35 | 12 | 52.47% | 1.221 |
| medium_adamw | 4: 32, 64, 128, 256 | Horizontal flip, rotation, translation, scale | AdamW, lr=0.0005, weight decay=0.0001, dropout=0.40 | 11 | 53.05% | 1.202 |

The **small** model had the highest validation accuracy, so it was selected before evaluating PrivateTest. Its PrivateTest accuracy was **61.33%**.

## Evaluation and analysis

The small model learned steadily through epoch 12. Its training accuracy at that point was 61.29%, close to its 60.04% validation accuracy, so this run does not show a large train/validation gap. The two deeper models were still below 50% training accuracy at epoch 12. Their lower validation scores suggest they need more training or different regularization and learning-rate settings; the current runs do not establish that deeper models are inherently worse.

PrivateTest recall by class, computed from the confusion matrix:

| Class | Images | Recall |
|---|---:|---:|
| Angry | 491 | 49.7% |
| Disgust | 55 | 14.5% |
| Fear | 528 | 23.3% |
| Happy | 879 | 86.2% |
| Sad | 594 | 64.3% |
| Surprise | 416 | 78.8% |
| Neutral | 626 | 57.2% |

Happy and surprise were recognized most reliably. Disgust and fear were difficult: disgust was often predicted as angry or sad, while fear was often predicted as sad. Angry and neutral images were also frequently confused with sad. Class imbalance is a likely factor, especially for disgust, which has only 55 PrivateTest images; these observations should be interpreted with that small sample in mind.

## Artifacts

- `results.json`: configurations, best validation results, final test accuracy, and confusion counts.
- `training_history.json`: per-epoch train and validation losses and accuracies.
- `experiment_comparison.png`: validation learning curves.
- `private_confusion_matrix.png`: row-normalized PrivateTest confusion matrix.
- `private_misclassified.png`: up to three PrivateTest errors per true class.
- `final_conv_filters.png`: mean absolute spatial kernels from the selected model's final convolution layer.
- `small_best.pt`, `medium_aug_best.pt`, `medium_adamw_best.pt`: best validation-loss checkpoints.

The plotted final convolution kernels average absolute weights across each filter's input channels to show one compact spatial map per output filter.
