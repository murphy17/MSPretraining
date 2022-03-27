import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from src.datamodule import MSDataModule
from src.model import MSTransformer

def main(hparams):
    dm = MSDataModule(**hparams)
    model = MSTransformer(**hparams)
    trainer = Trainer(
        gpus=hparams['num_gpus'],
        max_epochs=hparams['max_epochs'],
        precision=hparams['precision'],
        strategy=hparams['strategy']
    )
    trainer.fit(model, dm)
    
if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(add_help=False)
    
    # datamodule
    parser.add_argument('--hdf_path',type=str)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--train_val_split',type=float)
    parser.add_argument('--cdhit_threshold',type=float)
    parser.add_argument('--cdhit_word_length',type=int)
    parser.add_argument('--tmp_env',type=str)
    parser.add_argument('--num_workers',type=int)
    parser.add_argument('--random_state',type=int)

    # model
    parser.add_argument('--model_dim',type=int)
    parser.add_argument('--model_depth',type=int)
    parser.add_argument('--num_heads',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--dropout',type=float)
    parser.add_argument('--max_length',type=int)
    parser.add_argument('--use_cls_token',type=bool)

    # trainer
    parser.add_argument('--num_gpus',type=int)
    parser.add_argument('--max_epochs',type=int)
    parser.add_argument('--precision',type=int)
    parser.add_argument('--strategy',type=str)

    # cluster
    parser.add_argument('--num_nodes',type=int)
    parser.add_argument('--num_cpus',type=int)
    parser.add_argument('--conda_env',type=str)
    parser.add_argument('--time',type=str)

    # tensorboard
    parser.add_argument('--login_node',type=str)

    hparams = parser.parse_args()

    main(vars(hparams))