# main.py
from src.train import train
from src.utils import same_seeds
import os
# 设置随机种子以确保结果可复现
same_seeds(2023)


def main():
    config = {
        "model_type": "GAN",
        "batch_size": 24,
        "lr": 1e-5,
        "n_epoch": 100,
        "n_critic": 1,
        "z_dim": 100,
         "workspace_dir": '.', # define in the environment setting
    }

    train(config)


if __name__ == '__main__':
    main()