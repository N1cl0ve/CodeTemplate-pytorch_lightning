import argparse
import datetime

import pytorch_lightning as pl
import torch.cuda
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
import os
import pytorch_lightning.loggers.tensorboard

from utils import instantiate_from_config

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def get_parse(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-c", "--config", nargs="*", required=True)
    parser.add_argument("-r", "--reproducible", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=2024)
    parser.add_argument('--logtype', type=str, default="tensorboard", nargs="?", choices=["wandb", "tensorboard"])
    parser.add_argument("-p", "--project", help="name of new or path to existing project", default="Quantization")
    parser.add_argument("-d", "--debug", action="store_true")

    return parser


def prepare_task(opt, config, lightning_config):
    task = instantiate_from_config(config.task)
    task.learning_rate = lightning_config.learning_rate

    return task


def prepare_data(opt, config):
    data = instantiate_from_config(config.data)

    return data


def prepare_lightning(opt, config):
    def nondefault_trainer_args(opt):
        parser = argparse.ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        args = parser.parse_args([])
        return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)

    if "accelerator" not in trainer_config:
        if torch.cuda.is_available():
            trainer_config["accelerator"] = "gpu"
        else:
            raise NotImplementedError("Train only on gpu!")
    if "devices" not in trainer_config:
        trainer_config["devices"] = torch.cuda.device_count()
    if "accumulate_grad_batches" not in trainer_config:
        trainer_config["accumulate_grad_batches"] = 1
    n_gpus = trainer_config["devices"]
    accumulate_grad_batches = trainer_config["accumulate_grad_batches"]
    trainer_config["strategy"] = "ddp"
    trainer_config["precision"] = opt.precision
    print(f"Running on GPUs {n_gpus}")
    print(f"Training with precision {opt.precision}")
    print(f"Accumulate_grad_batches = {accumulate_grad_batches}")
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config["trainer"] = trainer_config

    if "base_learning_rate" in lightning_config:
        print("Using base_learning_rate & Configure learning rate According to batch_size!")
        bs, base_lr = config.data.params.batch_size[0], lightning_config.base_learning_rate
        learning_rate = accumulate_grad_batches * n_gpus * bs * base_lr
        lightning_config.learning_rate = learning_rate
    elif "learning_rate" in lightning_config:
        print("Using default learning_rate")
    else:
        raise NotImplementedError("Please set learning rate in config.lightning!")

    name = now + '_' + config.name
    logdir = os.path.join('logs', name)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    trainer_kwargs = dict()
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "project": opt.project,
                "name": name,
                "save_dir": os.path.join(os.getcwd(), logdir),
                "offline": opt.debug,
                "id": name,
            }
        },
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "name": "tensorboard",
                "save_dir": logdir,
            }
        },
    }
    default_logger_cfg = OmegaConf.create(default_logger_cfgs[opt.logtype])
    custom_logger_cfg = lightning_config.get("logger", OmegaConf.create())
    logger_cfg = OmegaConf.merge(default_logger_cfg, custom_logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    lightning_config["logger"] = logger_cfg

    default_callback_cfgs = {
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}-{train_loss:.4f}-{val_loss:.4f}",
                "verbose": True,
                "every_n_epochs": int(opt.check_val_every_n_epoch),
                "save_last": True,
                "save_top_k": -1,
            }
        }
    }
    default_callbacks_cfg = OmegaConf.create(default_callback_cfgs)
    custom_callbacks_cfg = lightning_config.get("callbacks", OmegaConf.create())
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, custom_callbacks_cfg)
    lightning_config["callbacks"] = callbacks_cfg
    setup_callback_cfg = {
        "setup_callback": {
            "target": "utils.SetupCallback",
            "params": {
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        }
    }
    setup_callback_cfg = OmegaConf.create(setup_callback_cfg)
    trainer_kwargs["callbacks"] = ([instantiate_from_config(setup_callback_cfg[k]) for k in setup_callback_cfg] +
                                   [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg])
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

    return lightning_config, trainer


if __name__ == "__main__":

    # TODO: 修改ExampleClassificationTask中的AlexNet，以匹配MNIST的输入图像与输出类别
    # TODO: 测试整个训练流程，确认日志文件的保存与tensorboard的可视化

    parser = get_parse()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    print(f"Current Workspace: {str(os.getcwd())}")
    print(f"Using Configs: {opt.config}")

    if opt.reproducible and opt.seed:
        seed_everything(seed=opt.seed)

    config = OmegaConf.load(opt.config[0])

    lightning_config, trainer = prepare_lightning(opt, config)
    task = prepare_task(opt, config, lightning_config)
    data = prepare_data(opt, config)

    trainer.fit(task, data)
    # trainer.save_checkpoint("")
