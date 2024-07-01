<p align="center">
    <h1 align="center">SEMI_SUPERVISED_STRESS_DETECTION</h1>
</p>
<p align="center">
    <em>Semi-supervised learning for wearable-based momentary stress detection in the wild</em>
</p>
<hr>

## 🔗 Quick Links

> - [📍 Overview](#-overview)
> - [📂 Repository Structure](#-repository-structure)
> - [🧩 Modules](#-modules)
> - [🚀 Getting Started](#-getting-started)

## 📍 Overview
Python (3.7) & Pytorch (1.7.0) implementation for paper: Semi-supervised learning for wearable-based momentary stress detection in the wild

```
@article{yu2023semi,
  title={Semi-supervised learning for wearable-based momentary stress detection in the wild},
  author={Yu, Han and Sano, Akane},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={7},
  number={2},
  pages={1--23},
  year={2023},
  publisher={ACM New York, NY, USA}
```

The `semi_supervised_stress_detection` project leverages semi-supervised learning to enhance stress detection accuracy and reduce labeling efforts. It offers a structured architecture for developing stress monitoring models, combining ECG and GSR signal analysis with deep learning techniques like autoencoders and pretrained models. Key components include configuration setup in `configs.py`, training orchestration in `train.py`, and task execution management in `run_tasks.py`. Through innovative training methods and feature extraction, this project provides a valuable tool for real-time stress analysis, aiding in understanding and managing stress levels effectively.

---

## 📂 Repository Structure

```sh
└── semi_supervised_stress_detection/
    ├── README.md
    └── src
        ├── configs.py
        ├── extract_embedding.py
        ├── model
        │   ├── __init__.py
        │   ├── cnn.py
        │   └── resnet.py
        ├── model_rl
        │   ├── __init__.py
        │   └── autoencoder.py
        ├── run_tasks.py
        ├── train.py
        └── utils
            └── dataset.py
```

---

## 🧩 Modules

<details closed><summary>src</summary>

| File                                                                                                                           | Summary                                                                                                                                                                                                                                                     |
| ---                                                                                                                            | ---                                                                                                                                                                                                                                                         |
| [configs.py](https://github.com/comp-well-org/semi_supervised_stress_detection/blob/master/src/configs.py)                     | Summary: In configs.py, defines dataset & training params for supervised stress detection. Includes batch sizes, LR, epochs, save paths, and pretrained model locations. Crucial for setting up supervised training in the stress detection system.         |
| [train.py](https://github.com/comp-well-org/semi_supervised_stress_detection/blob/master/src/train.py)                         | Code Summary:**`train.py` orchestrates training tasks utilizing configurable settings from `configs.py`. Facilitates model training and evaluation within the repository's stress detection architecture. Executed procedures streamline model development. |
| [run_tasks.py](https://github.com/comp-well-org/semi_supervised_stress_detection/blob/master/src/run_tasks.py)                 | Code Summary:**`run_tasks.py` orchestrates training tasks using Torch data loaders. Determines training mode based on configurations: supervised or semi-supervised. Loads data, initializes models, and conducts training accordingly.                     |
| [extract_embedding.py](https://github.com/comp-well-org/semi_supervised_stress_detection/blob/master/src/extract_embedding.py) | Code in `src/` trains stress detection models using semi-supervised learning for improved real-time stress monitoring. It enhances accuracy and reduces labeling efforts.                                                                                   |

</details>

<details closed><summary>src.utils</summary>

| File                                                                                                             | Summary                                                                                                                                                                                                                          |
| ---                                                                                                              | ---                                                                                                                                                                                                                              |
| [dataset.py](https://github.com/comp-well-org/semi_supervised_stress_detection/blob/master/src/utils/dataset.py) | Code Summary:** A script in `semi_supervised_stress_detection` repo that trains semi-supervised learning models using autoencoder architecture for stress detection, facilitating robust feature extraction for stress analysis. |

</details>

<details closed><summary>src.model</summary>

| File                                                                                                           | Summary                                                                                                                                                                                                                |
| ---                                                                                                            | ---                                                                                                                                                                                                                    |
| [resnet.py](https://github.com/comp-well-org/semi_supervised_stress_detection/blob/master/src/model/resnet.py) | Code snippet: `run_tasks.py`Summary: Orchestrates task execution, coordinating ML model training, feature extraction, and performance evaluation within the repository's structured machine learning pipeline.         |
| [cnn.py](https://github.com/comp-well-org/semi_supervised_stress_detection/blob/master/src/model/cnn.py)       | Code Summary:**The `model_conv1d` class combines ECG and GSR signal encoders, applies dropout and fully connected layers to classify stress levels. Pretrained weights are loaded based on the provided configuration. |

</details>

<details closed><summary>src.model_rl</summary>

| File                                                                                                                        | Summary                                                                                                                                                                                                                              |
| ---                                                                                                                         | ---                                                                                                                                                                                                                                  |
| [autoencoder.py](https://github.com/comp-well-org/semi_supervised_stress_detection/blob/master/src/model_rl/autoencoder.py) | Code snippet in `semi_supervised_stress_detection/` extracts embeddings using CNN and ResNet models to enhance semi-supervised stress detection. Key role: powering feature extraction for stress analysis in the parent repository. |

</details>

---

## 🚀 Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **Python**: `version 3.7`

### ⚙️ Installation

1. Clone the semi_supervised_stress_detection repository:

```sh
git clone https://github.com/comp-well-org/semi_supervised_stress_detection
```

2. Change to the project directory:

```sh
cd semi_supervised_stress_detection
```

3. Settings & Run:

Settings of augmentations, batch size, learning rates, etc. should be configured in `src/configs.py` file.

Please add your datasets in the `src/utils/dataset.py` file. For the labeled datasets, we aim to output the data format as `Sequence(s): Channels (C) x Length (L)` and `Label`; whereas we will have the sequences and augmentation views for the unlabeled set, as discussed in the paper.

Use `src/run_tasks.py` to run experiments.
