import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
import os
import glob

from websockets.frames import prepare_data


def get_parse(**parser_kwargs):

    # TODO: 确定要添加什么参数，一般来说是与任务有关超参数

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-c", "--config", nargs=1, required=True)
    parser.add_argument("-s", "--seed", type=int, default=2024)
    parser.add_argument('--logtype', type=str, default="tensorboard", nargs="?")
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()

    return parser


def prepare_model(opt, model_config):
    pass

def prepare_data(opt, data_config):
    pass

def prepare_lightning(opt, lightning_config):
    pass


if __name__ == "__main__":

    # TODO: 确定一下任务的命名规范，即日志文件的存储目录名
    # TODO: 确定一下trainer相关函数的实现，使用config文件指定各个参数

    parser = get_parse()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    print(f"Current Workspace: {str(os.getcwd())}")
    print(f"Using Configs: {opt.config}")

    config = OmegaConf.load(opt.config)
    name = config.name
    model_config = config.model
    data_config = config.data
    lightning_config = config.pop("lightning", OmegaConf.create())

    model = prepare_model(opt, model_config)
    data = prepare_data(opt, data_config)
    trainer = prepare_lightning(opt, lightning_config)

    trainer.fit(model, data)
    trainer.save_checkpoint("")
