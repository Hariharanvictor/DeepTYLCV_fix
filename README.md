# DeepTYLCV_fix
<h1 align="center">DeepTYLCV</h1>
<p align="center"><a href="https://balalab-skku.org/DeepTYLCV/">🌐 Webserver (CBBL-SKKU)</a> | <a href="https://1drv.ms/f/some_onedrive_link">🚩 Model & Dataset</a></p>

The official implementation of **DeepTYLCV: A deep learning framework for severity classification of Tomato Yellow Leaf Curl Virus using multi-scale Transformer–CNN fusion of PLM-based representations and conventional descriptors**

## TOC

This project is summarized into:
- [Installing environment](#installing-environment)
- [Preparing datasets](#preparing-datasets)
- [Configurations](#configurations)
- [Predicting models](#predicting-models)
- [Inferencing models](#inferencing-models)

## Installing environment
First, install [Miniconda](https://docs.anaconda.com/miniconda/) or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).  
Then create and activate the environment:

```bash
conda create -n deeptylcv python=3.10
conda activate deeptylcv

cd DeepTYLCV/
python -m pip install -r requirements.txt --no-cache-dir
(Optional) If you want to run the Inferencer (inference.py), please clone the optimized iFeatureOmegaCLI
 repository inside the src/ directory.
This version improves the performance of descriptor computation.

cd src/
git clone https://github.com/duongttr/iFeatureOmegaCLI
cd ..

Preparing datasets

DeepTYLCV uses open genomic and protein datasets derived from Tomato Yellow Leaf Curl Virus (TYLCV) samples.
Sequences are converted into Open Reading Frames (ORFs) and represented using three Protein Language Models (PLMs) — ESM-1, ESM-2, and ProtT5 — fused with Conventional Descriptor Features (CCDs) computed using iFeatureOmega.

You can:

Download the extracted feature datasets from OneDrive

Or generate your own features using the scripts in src/feature_extractor.py

Each sample is represented as:

PLM embeddings: ESM-2 (2560-dim), ESM-1 (1280-dim), ProtT5 (1024-dim)

Conventional features (CCDs): 394 or 887 selected features depending on the model version

Configurations

You can find configuration files in the configs/ directory:

config_DeepTYLCV_Hybrid.yaml — for Hybrid (NLP + CCD) mode

config_DeepTYLCV_NLP_only.yaml — for PLM-based mode only

Main parameters include:

dataset:
  dataset_root: Features/
  feature_1_name: ESM_2
  feature_2_name: ESM_1
  feature_3_name: ProtT5
  handcraft_name: CCD
...
model_config:
  drop: 0.3
  gated_dim: 256
  handcraft_dim: 394        # or 887 based on CCD version
  input_dim_1: 2560
  input_dim_2: 1280
  input_dim_3: 1024
  n_classes: 2
  num_heads_attn: 2
  num_heads_transformer: 2
  num_layers_transformers: 2
  num_mlp_layers: 4
...
trainer_config:
  batch_size: 128
  epochs: 100
  lr: 0.0001
  k_fold: 5
  loss_fn: focal
  threshold: 0.5
  output_path: checkpoints/DeepTYLCV_Hybrid

Predicting models

The DeepTYLCV_Predictor module in predictor.py allows you to run inference on extracted features:

from predictor import DeepTYLCV_Predictor
import yaml

config = yaml.safe_load(open('configs/config_DeepTYLCV_Hybrid.yaml'))

predictor = DeepTYLCV_Predictor(
    model_config=config['model_config'],
    ckpt_dir='/path/to/ckpt/dir',
    nfold=5,
    device='cuda'
)

# Predict a single sample
output = predictor.predict_one(
    f1=feature_1,  # ESM-2 (1, L, 2560)
    f2=feature_2,  # ESM-1 (1, L, 1280)
    f3=feature_3,  # ProtT5 (1, L, 1024)
    fccd=feature_ccd,  # CCD (1, 394 or 887)
    threshold=0.5
)


You can download pre-trained models and features from OneDrive
.

Note: The predictor expects extracted features. To automatically handle feature extraction, use the Inferencer described below.

Inferencing models

You can easily predict directly from FASTA files or sequence dictionaries using the Inferencer:

from predictor import DeepTYLCV_Predictor
from inference import Inferencer
import yaml

config = yaml.safe_load(open('configs/config_DeepTYLCV_Hybrid.yaml'))

predictor = DeepTYLCV_Predictor(
    model_config=config['model_config'],
    ckpt_dir='/path/to/ckpt/dir',
    nfold=5,
    device='cuda'
)

infer = Inferencer(predictor, device='cuda')

# Predict from FASTA file
outputs = infer.predict_fasta_file(
    fasta_file='/path/to/input.fasta',
    threshold=0.5,
    batch_size=4
)

# Predict from sequences directly
outputs_seq = infer.predict_sequences(
    data_dict={'TYLCV_seq_1': 'MSSSHIFIGETIGT', 'TYLCV_seq_2': 'CFGG'},
    threshold=0.5,
    batch_size=4
)

# Save results
infer.save_csv_file(outputs, 'DeepTYLCV_predictions.csv')


✨ DeepTYLCV provides an end-to-end deep learning pipeline — from feature extraction and model fusion to sequence-level prediction — to classify the severity of TYLCV infection using multi-scale PLM fusion and biological descriptors.
