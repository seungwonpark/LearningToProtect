import os
import time
import logging
import argparse

from nncrypt.train import train
from nncrypt.hparams import HParam
from nncrypt.writer import MyWriter
from nncrypt.data import create_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the run for logging")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    log_dir = os.path.join(hp.log.log_dir, args.name)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    writer = MyWriter(hp, log_dir)

    trainloader = create_dataloader(hp, True)
    valloader = create_dataloader(hp, False)

    train(args, trainloader, valloader, writer, logger, hp, hp_str)
