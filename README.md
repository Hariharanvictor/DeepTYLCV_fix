<h1 align="center">DeepTYLCV</h1>
<p align="center"><a href="https://balalab-skku.org/DeepTYLCV/">🌐 Webserver (CBBL-SKKU)</a> | <a href="https://zenodo.org/records/17636038">🚩 Models</a></p>

The official implementation of **DeepTYLCV: An interpretable and experimentally validated AI model for predicting virulence in tomato yellow leaf curl virus**

you can easily use DeepTYLCV tool with <a href="https://balalab-skku.org/DeepTYLCV/">🌐 Webserver (CBBL-SKKU)</a> or follow the below steps to train or inference the DeepTYLCV model.  

## TOC

This project is summarized into:

- [Installing environment](#installing-environment)
- [Preparing datasets](#preparing-datasets)
- [Configurations](#configurations)
- [Training models](#training-models)
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
## Preparing datasets

The training and test ORFs used for this project is deposited in the data/. We used 3 main PLM models: ESM, PTAB and PTBB, and optimal concatenated conventional descritpors extracted from iFeatureOmega. We already extracted features for all of them and optimal concantenated features are available in opt_CCD_features/ and PLM features can be downloaded from <a href="https://zenodo.org/records/17636038">Zenodo</a>.

## Configurations
You can find the configurations inside configs/ folder.
There are some parameters you should concentrate on:
```bash
dataset:
  conv_lmdb_path: /path/to/opt_CCD_features # mention the path to opt_CCDs features file
  data_root: /path/to/PLM/features # mention the directory to PLM features
  feature_list:
  - ESMEmbedder
  - ProtTransAlbertBFDEmbedder
  - ProtTransBertBFDEmbedder
  max_length: 363
  mean: false
  test_fasta_path: /path/to/test.fasta/file
  train_dir: /path/to/train/5-fold/directory
  tran_nlp: true
...
trainer:
  batch_size: 16
  device: cpu
  epochs: 100
  lr: 0.0001
  num_workers: 4
  output_path: /path/to/save_results # mention path to save results
```
## Training Models
To reconstruct the results from the paper, you can run the following command to train the models:
```bash
python train.py --config_path configs/config_DeepTYLCV.yaml --save_config
```
## Inferencing models
You can easily predict directly from FASTA files or sequence dictionaries using the Inferencer by giving path to the model files. model files can be downloaded from <a href="https://zenodo.org/records/17510705">Models</a>:
```bash
from predictor import DeepTYLCV_Predictor
from inference import Inferencer
import yaml

config = yaml.safe_load(open('configs/config_DeepTYLCV.yaml'))

predictor = DeepTYLCV_Predictor(
    model_config=config['model'],
    ckpt_dir='/path/to/ckpt/dir',
    nfold=5,
    device='cuda'
)

infer = Inferencer(predictor, scaler_path='/path/to/scaler/model' device='cuda')

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
