# DeepTYLCV_fix
<h1 align="center">DeepTYLCV</h1>
<p align="center"><a href="https://balalab-skku.org/DeepTYLCV/">🌐 Webserver (CBBL-SKKU)</a> | <a href="https://zenodo.org/records/17510705">🚩 Model files</a></p>

The official implementation of **DeepTYLCV: An interpretable and experimentally validated AI model for predicting virulence in tomato yellow leaf curl virus**


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
**Inferencing models**
You can easily predict directly from FASTA files or sequence dictionaries using the Inferencer by giving path to the model files. model files can be downloaded from <a href="https://zenodo.org/records/17510705">🚩Models</a>:
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
