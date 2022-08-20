# Traffic Speed Imputation with Spatio-Temporal Attentions and Cycle-Perceptual Training
> Accepted to CIKM'22.

## Requirements
- PyTorch 1.10.0
- PyTorch Geometric 2.0.2

## Usage
- Download datasets (note that Chengdu dataset is from Didi GAIA, which is not public) from [Google Drive](https://drive.google.com/file/d/1X5muVWGWCPRs9k1fskOBnJxkDh5D_ZSu/view?usp=sharing).
- Download pretrained models from [Google Drive](https://drive.google.com/file/d/19NWZgoRIuuGN36xtrugLbleGN0ykm_mo/view?usp=sharing).
```
> cd STCPA
> conda env create -f stcpa_env.yaml
> conda activate stcpa
> pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
# Test on New York datasets.
> python test_stcpa_nyc.py
```
