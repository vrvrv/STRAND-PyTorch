# STRAND-PyTorch

[R version](https://github.com/emauryg/STRAND_R)

# Train
```bash
# trainer.gpus=0 : Only uses CPU
# trainer.gpus=1 : Can use the single GPU
python run.py experiment=ts model.rank=10 trainer.gpus=1
```