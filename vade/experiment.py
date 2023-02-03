import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import make_data
from model import Model
from utils.file import save_file
from utils.nn_utils import make_trainer


def main(config):
    seed = config.__dict__.pop("seed")
    save_file(config, os.path.join(config.save_dpath, "args.pkl"))
    pl.seed_everything(seed)
    train_loader, val_loader, test_loader = make_data(config.data_dpath, config.batch_size, config.n_workers)
    model = Model(config.input_dim, config.latent_dim, config.n_components, config.lr)
    trainer = make_trainer(config.save_dpath, seed, config.n_epochs)
    if config.is_test:
        trainer.test(model, test_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dpath", type=str, default="data")
    parser.add_argument("--save_dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=784)
    parser.add_argument("--latent_dim", type=int, default=200)
    parser.add_argument("--n_components", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--is_test", action="store_true")
    main(parser.parse_args())