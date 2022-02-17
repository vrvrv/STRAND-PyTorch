# STRAND-PyTorch

[R version](https://github.com/emauryg/STRAND_R)

# Get Started

## Conda env
```bash
conda create -n STRAND python=3.8 -y
conda activate STRAND
```

```bash
git clone https://github.com/vrvrv/STRAND-PyTorch.git
cd STRAND-PyTorch
pip install -r requirements.txt
```

## Preparing Data
You must put the data in [data](data/) directory. If the name of your dataset is `pcawg`, there must be two pickled(`.pkl`)
files at `data/pcawg`, `snv.pkl` and `feature.pkl`.

## Running STRAND
There are several prescribed [configurations](configs/experiment) to run our model with different settings.
For example, if you want to run our model on `PCAWG` dataset and want to model with 10 signatures accelerated by `GPU`,
enter the following sentence.

```bash
# trainer.gpus=0 : Only uses CPU
# trainer.gpus=1 : Can use the single GPU
python run.py experiment=pcawg model.rank=10 +trainer.gpus=1
```

Above code brings training configuration and model hyperparameters from [configs/experiment/pcawg.yaml](configs/experiment/pcawg.yaml).
The `YAML` file looks
```yaml
# @package _global_

data_name: pcawg

model:
  e_iter: 15
  tf_max_steps: 200
  laplace_approx_conf:
    lr: 0.01
    max_iter: 2000
    batch_size: 512
  nmf_max_iter: 100

trainer:
  max_epochs: 30

logger:
  _target_: pytorch_lightning.loggers.wandb.WandbLogger
  project: "STRAND-Lightning-pcawg"
  save_dir: "."
  group: ${data_name} # pcawg
  name: ${data_name}_rank_${model.rank} # pcawg_rank_10
```

You can test the code quickly by using [configs/experiment/pcawg.yaml](configs/experiment/pcawg_fast_run.yaml).
## Running Simulations

### Generating simulated data
```bash
# id (int) : ID of simulated data
# nbinom (bool) : The distribution of count tensor
# disp_param (float) : dispersion parameter -> Variance = Mean + Mean ** 2 / disp_param
python simul_data_generator.py --id=1 --nbinom=True --disp_param=50
```
It generates the data with number of samples `[50, 100, 1000, 2000]`, number of mutations per sample `[50, 100, 1000, 2000]`
with underlying `[5, 10, 20, 30]`.

### Traing `STRAND` on simulated data
```
# sid (int) : ID of simulated data
# n : number of samples
# m : number of mutations per sample
# true_rank : true number of signatures of the generated data
# rank : the rank of model fitting

python run_sim.py experiment=simulation sid=1 n=50 m=100 true_rank=10 model.rank=5 trainer.gpus=1
```
It saves the best checkpoint at `logs/runs/simulation_1/rank_5_true_rank_10_n_50_m_100`

