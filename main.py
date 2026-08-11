import os
import logging
import hydra
import shutil

from trainer import train
from inference.inference import inference


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Host : {os.uname()[1]}")

    shutil.copy2(".hydra/config.yaml", "config.yaml")

    if cfg.mode == "train":
        return train(cfg)
    elif cfg.mode == "inference":
        return inference(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode=}")


if __name__ == "__main__":
    main()
