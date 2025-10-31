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
