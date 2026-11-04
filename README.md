# seng402 - Finding Small Things in Big Images

## Description
Pipeline for generating and pre-processing a dataset of small targets hidden in noise, then training a CNN to detect these targets. After training, a heat-map visualisation of where targets are predicted to be is produced.

The project also supports:

- Training with an iterable dataset that uses all segments from each generated image.
- Segment-size and overlap sweeps, ranked by validation F1 score.
- Generalisation tests on target sizes, shapes, and mixtures that differ from the training distribution.
- Transfer learning from the synthetic target detector to the breast-cancer cell segmentation dataset.

## Setup Instructions
#### **1) Create & activate virtual environment**
In your local directory:<br>  
python3 -m venv .venv  
source .venv/bin/activate

#### **2) Upgrade pip**
python -m pip install --upgrade pip

#### **3) Install dependencies**
python -m pip install -r requirements.txt

If the environment gets into a bad state, delete `.venv/` and run these steps again.

## Usage
All main workflows are configured in the `run_config` dictionary near the bottom of
`train_iterable.py`. Run the default synthetic training workflow with:

```bash
python train_iterable.py
```

The default workflow trains the synthetic detector, saves its checkpoint as
`synthetic_watermark_cnn.pt`, and writes a prediction heat map to
`prediction_visual_new.png`.

### Synthetic training

Update the `training`, `data`, `targets`, and `visualization` sections of
`run_config` to change the number of epochs, batch size, learning rate, image
size, segment size, overlap, target distribution, or visualisation settings.
Training reports loss, accuracy, precision, recall, and F1 for both the training
and validation distributions.

### Segment sweep

Set the following option before running `train_iterable.py`:

```python
"mode": {
    "run_sweep": True,
    "run_medical_transfer": False,
},
```

The sweep evaluates the configured combinations of segment sizes and overlaps,
repeats each run with different seeds, ranks the results by validation F1, and
writes:

- `segment_sweep_results.csv`
- `segment_sweep_results.png`

### Generalisation tests

Set `"generalisation": {"enabled": True, ...}` in `run_config`. The configured
cases can test different target sizes, shapes (`"circle"`, `"square"`, or
`"mixed"`), and mixing modes. Results are written to
`generalisation_results.csv`.

### Medical transfer learning

The medical workflow uses the synthetic checkpoint to initialise a spatial
segmentation model. It compares:

1. A synthetic classifier baseline.
2. A model with a frozen synthetic encoder and trained segmentation head.
3. A fine-tuned model.

Set:

```python
"mode": {
    "run_sweep": False,
    "run_medical_transfer": True,
},
```

The default medical configuration expects the breast-cancer dataset at
`data/breast-cancer-cell-segmentation/` with `Images/` and `Masks/`
subdirectories. The dataset can be downloaded by calling
`download_breast_cancer_dataset()` from `train_iterable.py`, or obtained
separately through Kaggle. The medical workflow writes model checkpoints and a
prediction overlay into `transfer_outputs/`.


## Recreating The Environment

This project is set up to be recreated from scratch with the same steps every time:

1. Remove any existing `.venv/` folder.
2. Create a new virtual environment with `python3 -m venv .venv`.
3. Activate it and run `python -m pip install --upgrade pip`.
4. Install packages with `python -m pip install -r requirements.txt`.

If Torch still fails to import, try using Python 3.12 or 3.11 for the virtual environment, since Torch wheels can be sensitive to the Python version.

## Project Structure

- `image_generation.py`: Synthetic image generation, target embedding, target-shape handling, and image segmentation.
- `pytorch_cnn.py`: CNN model definitions for synthetic detection and spatial segmentation.
- `train_detection.py`: Legacy training loop using one random segment per image.
- `train_iterable.py`: Iterable-dataset training using all segments per image, validation, sweeps, generalisation tests, checkpointing, and heat-map visualisation.
- `transfer_learning.py`: Medical dataset discovery, patch extraction, transfer-learning stages, evaluation, and prediction overlays.
- `tests/`: Automated tests for sweep configuration and result ranking.
- `data/breast-cancer-cell-segmentation/`: Optional medical dataset location.
- `*.csv`, `*.png`, and `*.pt`: Example results, visualisations, and saved model checkpoints.

## Author
**Finn Brown**