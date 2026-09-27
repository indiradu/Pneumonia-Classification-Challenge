# ENG 2440 Assignment 1: Lung-opacity classification on RSNA chest radiographs

Student: Indira Duisembayeva (25011125)

`25011125_ENG2440_A1.ipynb` is the fully executed notebook. It covers:
- patient-disjoint splits
- preprocessing and augmentation
- trivial and logistic baselines
- an ImageNet-pretrained ResNet18 (fixed-feature phase, then fine-tuning of `layer4`)
- evaluation, operating thresholds and calibration
- subgroup analysis and Grad-CAM
- Bonus question 2 (border-masking stress test with a lung-field control)

> Learning exercise only, not a clinical diagnostic system.

## Data (not included in this repository)

The RSNA images and label/mapping files are **not** uploaded, as the assignment requires.
Place them next to the notebook like this:

```
data/images/*.dcm            # 25,684 RSNA DICOM files
assignment1_labels.csv       # RSNA annotation table (patientId, x, y, width, height, Target, ...)
rsna_to_nih_mapping.csv      # supplied RSNA-to-NIH mapping (patientId, nih_patient_id, nih_image_id)
```

## Environment

- Python **3.13.5** was used to produce the results. **Python ≥ 3.12 is required**, because two cells (Part A counts and Part B split sizes) use nested double quotes inside f-strings, which older Python versions reject.
- Package versions: see `requirements.txt`. matplotlib must be **≥ 3.9**, because the Bonus 2 box plot uses `boxplot(label=...)`.
- Hardware used for the reported results: Apple Silicon Mac (arm64), **CPU only**. The notebook uses CUDA automatically if it is available. GPU runs can differ slightly in the last decimals because some GPU kernels are non-deterministic.
- Internet access is needed on the first run to download the torchvision ResNet18 ImageNet weights (`ResNet18_Weights.DEFAULT`, which for ResNet18 is the `IMAGENET1K_V1` checkpoint, `resnet18-f37072fd.pth`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook 25011125_ENG2440_A1.ipynb    # then Kernel → Restart & Run All
```

## Random seeds

| Component | Seed |
|---|---|
| Train/validation/test split (`GroupShuffleSplit`, grouped by `nih_patient_id`) | `random_state=7` |
| Augmentation before/after figure | `torch.manual_seed(7)` (the intensity range printed in the first Part C cell is not seeded and can vary between runs; it is not used anywhere else) |
| Small-CNN class-imbalance ablation | `torch.manual_seed(7)` |
| Logistic baseline | `torch.manual_seed(7)` |
| ResNet18 runs (`np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`; these also fix `DataLoader` shuffling and augmentation, `num_workers=0`) | 7 (headline), 8, 9; unweighted ablation 7 |
| Example and Grad-CAM image selection | `np.random.default_rng(7)`, `DataFrame.sample(random_state=7)` |

## Expected key results (to check a reproduction)

- Splits: train 15,278 / validation 5,250 / test 5,156 examinations; 11,171 source patients; disjointness assertions pass.
- ResNet18 validation PR-AUC over seeds 7/8/9: 0.635 ± 0.002.
- Test set, threshold 0.5:

| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| ResNet18 | 0.850 | 0.614 | 0.710 | 0.821 |
| Logistic baseline | 0.513 | 0.230 | – | – |

- Frozen thresholds (chosen on validation): screening 0.2209, high-specificity 0.6780.
