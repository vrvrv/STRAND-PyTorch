# STRAND-PyTorch

[R version](https://github.com/emauryg/STRAND_R)

# Get Started
There are several prescribed [configurations](configs/experiment) to run our model with different settings.
For example, if you want to run our model on `PCAWG` dataset and want to model with 10 signatures accelerated by `GPU`,
enter the following sentence.

```bash
# trainer.gpus=0 : Only uses CPU
# trainer.gpus=1 : Can use the single GPU
python run.py experiment=pcawg model.rank=10 trainer.gpus=1
```

It brings training configuration and model hyperparameters from [configs/experiment/pcawg.yaml](configs/experiment/pcawg.yaml).
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
  project: "STRAND-Lightning-Simulation_pcwag"
  save_dir: "."
  group: ${data_name} # pcawg
  name: ${data_name}_rank_${model.rank} # pcawg_rank_10
```
