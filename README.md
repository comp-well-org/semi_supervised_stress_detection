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

Settings of augmentations, batch size, learning rates, etc. should be configured in `src/configs.py` file.

Please add datasets in the `src/utils/dataset.py` file. For the labeled datasets, we aim to output the data format as `Sequence(s): Channels (C) x Length (L)` and `Label`; whereas we will have the sequences and augmentation views for the unlabeled set, as discussed in the paper.

Use `src/run_tasks.py` to run experiments.

