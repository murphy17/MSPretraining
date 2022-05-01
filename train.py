import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.datamodule import MSDataModule
from src.model import MSTransformer
# from pytorch_lightning.plugins.training_type.ddp import DDPPlugin

def main(hparams):
    seed_everything(hparams['random_state'], workers=True)
    dm = MSDataModule(**hparams)
    model = MSTransformer(**hparams)
    trainer = Trainer(
        gpus=hparams['num_gpus'],
        num_nodes=hparams['num_nodes'],
        max_epochs=hparams['max_epochs'],
        precision=hparams['precision'],
        strategy=hparams['strategy'],
#         strategy=DDPPlugin(find_unused_parameters=False),
        callbacks=[
            EarlyStopping(
                monitor=hparams['es_monitor'],
                mode=hparams['es_mode'],
                patience=hparams['es_patience']
            )
        ]
    )
    trainer.fit(model, dm)
    
if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(add_help=False)
    
    # datamodule
    parser.add_argument('--hdf_path',type=str)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--train_split',type=float)
    parser.add_argument('--val_split',type=float)
    parser.add_argument('--cdhit_threshold',type=float)
    parser.add_argument('--cdhit_word_length',type=int)
    parser.add_argument('--tmp_env',type=str)
    parser.add_argument('--num_workers',type=int)
    parser.add_argument('--random_state',type=int)

    # model
    parser.add_argument('--model_dim',type=int)
    parser.add_argument('--model_depth',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--dropout',type=float)

    # trainer
    parser.add_argument('--num_gpus',type=int)
    parser.add_argument('--max_epochs',type=int)
    parser.add_argument('--precision',type=int)
    parser.add_argument('--strategy',type=str)
    parser.add_argument('--es_monitor',type=str)
    parser.add_argument('--es_mode',type=str)
    parser.add_argument('--es_patience',type=int)

    # cluster
    parser.add_argument('--num_nodes',type=int)
    parser.add_argument('--num_cpus',type=int)
    parser.add_argument('--conda_env',type=str)
    parser.add_argument('--time',type=str)

    # tensorboard
    parser.add_argument('--login_node',type=str)

    hparams = parser.parse_args()

    main(vars(hparams))