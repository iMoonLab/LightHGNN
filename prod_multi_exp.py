import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from prod_train import exp
from utils import MultiExpMetric


@hydra.main(config_path=".", config_name="prod_config", version_base="1.1")
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))
    res_all = MultiExpMetric()
    for seed in range(5):
        res = exp(seed, cfg)
        res_all.update(res)
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info(res_all)


if __name__ == "__main__":
    main()
