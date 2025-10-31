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
```
Next, install the required dependencies:
```bash
cd DeepTYLCV/
python -m pip install -r requirements.txt --no-cache-dir
```
**Preparing datasets**

DeepTYLCV uses Open Reading Frames(ORF) originated from Genomes sequences. ORF can be found in the data/ directory. In this project, we used 3 Protein Language models(PLM): ESM, ProtransAlbertBFD(PTAB), ProtransBertBFD(PTBB), and optimal concatenated conventional descriptors features(optCCDS) extracted from iFeatureOmega. We already extracted features for all of them and they can be downloaded from <a href="https://balalab-skku.org/DeepTYLCV/">OneDrive</a>.
Sequences are converted into Open Reading Frames (ORFs) and represented using three Protein Language Models (PLMs) — ESM-1, ESM-2, and ProtT5 — fused with Conventional Descriptor Features (CCDs) computed using iFeatureOmega.

**Configurations**

You can find configuration files in the configs/ directory:


Main parameters include:
```bash
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
```
**Inferencing models**
You can easily predict directly from FASTA files or sequence dictionaries using the Inferencer:
```bash
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
```
DeepTYLCV provides an end-to-end deep learning pipeline — from feature extraction and model fusion to sequence-level prediction — to classify the severity of TYLCV infection using multi-scale PLM fusion and biological descriptors.
