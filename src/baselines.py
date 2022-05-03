import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import auroc, accuracy
from collections import defaultdict

class SequenceModel(LightningModule):
    def __init__(
        self,
        output_dim,
        model_dim,
        model_depth,
        num_residues,
        dropout,
        lr
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.model_depth = model_depth
        self.num_residues = num_residues
        self.output_dim = output_dim
        self.dropout = dropout
#         self.output_weights = output_weights
        self.lr = lr
        self.name = None
        
        if num_residues is not None:
            self.embedding = nn.Embedding(
                num_embeddings=num_residues,
                embedding_dim=model_dim,
                padding_idx=0
            )

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),lr=self.lr)
        return opt
    
    def unbatch(self, batch):
        batch_size = len(batch['sequence'])
        x = batch['x']
        x_mask = batch['x_mask']
        y = batch['y']
        return (x, x_mask, y), batch_size
    
    def step(self, batch, batch_idx, *, step):
        (x, x_mask, y), batch_size = self.unbatch(batch)
        
        y_pred = self(x, x_mask)
        
        metrics = {}
        
        if self.output_dim == 1:
            y_pred = y_pred.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
        else:
            loss = F.cross_entropy(y_pred, y)
            
        auc = auroc(y_pred, y, average='macro')
        metrics['auc'] = auc
        
        if step != 'predict':
            self.log(f'{step}_loss',loss,batch_size=batch_size)

            for m in metrics:
                self.log(
                    f'{step}_{m}_{self.name}',
                    metrics[m],
                    batch_size=batch_size,
                )

        return loss, metrics, y_pred
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx, step='train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, step='val')
            
    def test_step(self, batch, batch_idx):
        _, metrics, _ = self.step(batch, batch_idx, step='test')
        return metrics
            
    def predict_step(self, batch, batch_idx):
        _, _, y_pred = self.step(batch, batch_idx, step='predict')
        y_pred = torch.sigmoid(y_pred)
        return y_pred

class CNNModel(SequenceModel):
    def __init__(
        self,
        kernel_size,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        
        encoder_layers = []
        in_dim = self.model_dim
        out_dim = self.model_dim
        for i in range(self.model_depth-1):
            out_dim = in_dim // 2
            drop = nn.Dropout(self.dropout)
            conv = nn.Conv1d(
                in_dim,
                out_dim,
                self.kernel_size,
                padding=self.kernel_size//2
            )
            norm = nn.BatchNorm1d(out_dim)
            relu = nn.LeakyReLU(0.2, inplace=True)
            pool = nn.MaxPool1d(2,2)
            encoder_layers += [drop, conv, norm, relu, pool]
            in_dim = in_dim // 2
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.classifier = nn.Linear(out_dim, self.output_dim)
        
    def forward(self, x, x_mask):
        x = self.embedding(x)
        x = x.swapdims(1,2)
        x = self.encoder(x)
        x = x.sum(-1) / x_mask.sum(-1).view(-1,1)
        x = self.classifier(x)
        return x
    
class LinearModel(SequenceModel):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs,
            model_depth=None,
            dropout=None
        )
        
        self.classifier = nn.Linear(self.model_dim, self.output_dim)
        
    def forward(self, x, x_mask):
        x = self.embedding(x)
        x = x.swapdims(1,2)
        x = x.sum(-1) / x_mask.sum(-1).view(-1,1)
        x = self.classifier(x)
        return x


from sequence_models.pretrained import load_model_and_alphabet
from sequence_models.structure import Attention1d

class ESMAttention1d(nn.Module):
    """Outputs of the ESM model with the attention1d"""
    def __init__(self, d_embedding, d_out):
        super(ESMAttention1d, self).__init__()
        self.attention1d = Attention1d(in_dim=d_embedding) # ???
        self.linear = nn.Linear(d_embedding, d_embedding)
        self.relu = nn.ReLU()
        self.final = nn.Linear(d_embedding, d_out)
    
    def forward(self, x, input_mask):
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.relu(self.linear(x))
        x = self.final(x)
        return x
    
# class ESMAttention1d(nn.Module):
#     """Outputs of the ESM model with averaging"""
#     def __init__(self, max_length, d_embedding, d_out):
#         super().__init__()
#         self.linear = nn.Linear(d_embedding, d_embedding)
#         self.relu = nn.ReLU()
#         self.final = nn.Linear(d_embedding, d_out)
    
#     def forward(self, x, input_mask):
#         x = (x*input_mask.unsqueeze(-1)).sum(1) / input_mask.unsqueeze(-1).sum(1)
#         x = self.relu(self.linear(x))
#         x = self.final(x)
#         return x

class CARPModel(SequenceModel): 
    def __init__(
        self,
        fixed_weights,
        **kwargs
    ):
        super().__init__(
            model_dim=128,
            model_depth=None,
            num_residues=None,
            dropout=None,
            **kwargs
        )

        model, self.collater = load_model_and_alphabet('carp_600k')

        self.encoder = model.embedder.requires_grad_(not fixed_weights)
        self.classifier = ESMAttention1d(
            self.model_dim, 
            self.output_dim
        )

    def unbatch(self, batch):
        batch_size = len(batch['sequence'])
        x, = self.collater([[s] for s in batch['sequence']])
        x = x.to(self.device)
        x_mask = batch['x_mask']
        y = batch['y']
        return (x, x_mask, y), batch_size
        
    def forward(self, x, x_mask):
        x = self.encoder(x, x_mask.unsqueeze(-1))
        x = self.classifier(x, x_mask)
        return x
    
from esm.pretrained import esm1b_t33_650M_UR50S

# Load ESM-1b model
# model, alphabet = esm1b_t33_650M_UR50S()
# batch_converter = alphabet.get_batch_converter()

class ESMModel(SequenceModel): 
    def __init__(
        self,
        fixed_weights,
        **kwargs
    ):
        super().__init__(
            model_dim=128,
            model_depth=None,
            num_residues=None,
            dropout=None,
            **kwargs
        )

        model, self.collater = load_model_and_alphabet('carp_600k')

        self.encoder = model.embedder.requires_grad_(False)
        self.encoder.eval()
        
        self.classifier = ESMAttention1d(
            self.model_dim, 
            self.output_dim
        )

    def unbatch(self, batch):
        batch_size = len(batch['sequence'])
        x, = self.collater([[s] for s in batch['sequence']])
        x = x.to(self.device)
        x_mask = batch['x_mask']
        y = batch['y']
        return (x, x_mask, y), batch_size
        
    def forward(self, x, x_mask):
        x = self.encoder(x, x_mask.unsqueeze(-1))
        x = self.classifier(x, x_mask)
        return x
    
from .model import MSTransformer

class MSModel(SequenceModel):
    def __init__(
        self,
        checkpoint,
        fixed_weights,
        naive,
        **kwargs
    ):
        super().__init__(
            model_depth=None,
            num_residues=None,
            dropout=None,
            **kwargs
        )

        self.encoder = MSTransformer.load_from_checkpoint(checkpoint)
        if naive:
            self.encoder = MSTransformer(**dict(self.encoder.hparams))
        self.encoder = self.encoder.x_encoder
        self.encoder.requires_grad_(not fixed_weights)

        self.classifier = ESMAttention1d(self.model_dim, self.output_dim)

    def forward(self, x, x_mask):
        x = self.encoder(x, x_mask.unsqueeze(-1))
        x = self.classifier(x, x_mask)
        return x

import numpy as np

class CARPPretextModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model, self.collater = load_model_and_alphabet('carp_600k')
        self.alphabet = self.collater.tokenizer.alphabet
        self.alphabet = np.array(list(self.alphabet))

    def forward(self, x, x_mask):
        y = self.model(x, x_mask.unsqueeze(-1))
        return y

    def predict(self, masked_sequences):
        x, = self.collater([[s] for s in masked_sequences])
        x_mask = x != 27
        x_pred = self(x, x_mask)
        x_pred = x_pred.argmax(-1)
        x_pred[~x_mask] = 27
        x_pred = x_pred.detach().cpu().numpy()
        x_pred = self.alphabet[x_pred]
        x_pred = [''.join(_) for _ in x_pred]
        return x_pred