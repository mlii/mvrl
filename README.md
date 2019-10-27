# Multi-View Reinforcement Learning

This repo includes the poster and experiments for the paper [Multi-View Reinforcement Learning](https://arxiv.org/abs/1910.08285).

## Requirements

Test with python3.6.1 (might work with later versions).
Install all requirements by

`pip install -r requirements.txt`

## Code structure

### data

- `extract.py/sh`: Sample the data used in model learning

### models

`vrnn.py`: Define the rnn structure.
`vae.py`: Define the vae structure.

### the rest

- `config.py`: Define the env.
- `env.py`: Preprocess the environment used in the model learning process.
- `train.py`: Train the multi-view model.
- `utils.py` and `ops.py`: Define many functions to simplify the multi-view model training code.
- `wrappers.py`: Define the transformer for each view.

## How to run

Download the data from [here](https://drive.google.com/file/d/1EfoQzGIYwLFvaxEZo-Fsa8oy2GKqn2_t/view?usp=sharing), extract the file, and put the extracted folder in `./data`.

You can then run
```
python train.py --model-dir checkpoint/model --data-dir data/record --view transposed --gpu 0
```

The output and training log is saved at `checkpoint/model`, the data is loaded from `data/record`, the view choice is `transposed` and the selected gpu is 0.


## Citation

If you found it helpful, consider citing the following paper:

```
@article{li2019multi,
  title={Multi-View Reinforcement Learning},
  author={Li, Minne and Wu, Lisheng and Ammar, Haitham Bou and Wang, Jun},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
