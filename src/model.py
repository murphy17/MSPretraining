import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from sequence_models.convolutional import ByteNet, ByteNetLM, MaskedConv1d

from .constants import MSConstants
C = MSConstants()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, stride: int = 1, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[offset:offset+x.size(1)*stride:stride].unsqueeze(0)
        return self.dropout(x)
    
# class MSDecoder(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         model_dim,
#         output_dim
#     ):
#         super().__init__()
#         self.input_dim = input_dim
#         self.model_dim = model_dim
#         self.output_dim = output_dim
#         self.conv1 = MaskedConv1d(
#             in_channels=input_dim,
#             out_channels=model_dim,
#             kernel_size=2 # residues -> bonds
#         )
#         self.conv1.padding = 0 # residues -> bonds
#         self.relu1 = nn.ReLU()
#         self.conv2 = MaskedConv1d(
#             in_channels=model_dim,
#             out_channels=model_dim,
#             kernel_size=1
#         )
#         self.relu2 = nn.ReLU()
#         self.conv3 = MaskedConv1d(
#             in_channels=model_dim,
#             out_channels=output_dim,
#             kernel_size=1
#         )
    
#     def forward(self, x, input_mask):
#         x = self.conv1(x, input_mask)
#         x = self.relu1(x)
#         x = self.conv2(x, input_mask[:,1:])
#         x = self.relu2(x)
#         x = self.conv3(x, input_mask[:,1:])
#         return x
    
class MSTransformer(pl.LightningModule):
    def __init__(
        self,
        model_dim,
        model_depth,
        lr,
        dropout, 
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.residues = C.alphabet
        self.ions = C.ions
        self.parent_charges = range(C.min_charge, C.max_charge + 1)
        self.fragment_charges = range(C.min_frag_charge, C.max_frag_charge + 1)
        self.losses = C.losses
        
        self.input_dim = len(self.residues)
        self.condition_dim = 2
        self.model_dim = model_dim
        self.output_dim = (
            len(self.ions), 
            len(self.fragment_charges),
            len(self.losses)
        )
        self.model_depth = model_depth
        self.dropout = dropout
        self.lr = lr
        
        self.embed_dim = 8
        self.kernel_size = 5
        self.r = 128
        self.padding_idx = 0
        self.masking_idx = self.input_dim
        
        self.x_encoder = ByteNet(
            n_tokens=self.input_dim + 1,
            d_embedding=self.embed_dim,
            d_model=self.model_dim,
            n_layers=self.model_depth,
            kernel_size=self.kernel_size,
            r=self.r,
            padding_idx=self.padding_idx, 
            causal=False,
            dropout=self.dropout,
            activation='gelu'
        )

        self.y_encoder = ByteNet(
            n_tokens=1, # unused
            d_embedding=np.prod(self.output_dim)+self.condition_dim,
            d_model=self.model_dim,
            n_layers=self.model_depth,
            kernel_size=self.kernel_size,
            r=self.r,
            padding_idx=self.padding_idx, 
            causal=False,
            dropout=self.dropout,
            activation='gelu'
        )
        self.y_encoder.embedder = nn.Identity()
        
        # final AA classifier should be VERY SIMPLE!
        self.conv1 = MaskedConv1d(
            self.model_dim * 2, 
            self.model_dim,
            kernel_size=1
        )
        self.relu = nn.ReLU()
        self.conv2 = MaskedConv1d(
            self.model_dim, 
            self.input_dim,
            kernel_size=1
        )

    def forward(self, x, y, c, input_mask):
        input_mask = input_mask.unsqueeze(-1)

        x = self.x_encoder(x, input_mask=input_mask)
        
        y = y.flatten(2)
        c = c.unsqueeze(1).expand(-1, y.shape[1], -1)
        y = torch.cat([y,c],-1)
        y = self.y_encoder(y, input_mask=input_mask)
#         y = x

        z = torch.cat([x,y],-1)
        z = self.conv1(z, input_mask=input_mask)
        z = self.relu(z)
        x_pred = self.conv2(z)

        return x_pred

    def step(self, batch, step):
        batch_size = batch['x'].shape[0]
        max_length = batch['x'].shape[1]

        x = batch['x']
        padding_mask = batch['x_mask']

        c = torch.stack([
            batch['charge'],
            batch['collision_energy']
        ],-1).float()

        y = batch['y']

        # this will affect longer spectra more...?
#         y_mask = torch.rand_like(y) < self.spectrum_dropout
#         y[y_mask] = 0
        
        # normalize spectrum; for domain reasons, but also fixes dropout ^^ ?
        y = y / y.flatten(1).sum(-1).view(-1,1,1,1,1)
        
        y_pad = torch.zeros(batch_size,1,*self.output_dim,device=self.device)
        y = torch.cat([y,y_pad],1)

        # to start, just a single AA
        masking_idx = torch.multinomial(padding_mask.float(), 1).squeeze()
        x_mask = torch.zeros_like(padding_mask,dtype=torch.bool)
        x_mask[range(batch_size),masking_idx] = 1
        x_masked = x.clone()
        x_masked[x_mask] = self.masking_idx

        x_pred = self(x_masked, y, c, padding_mask)

        xent = F.cross_entropy(
            x_pred[range(batch_size),masking_idx],
            x[range(batch_size),masking_idx]
        )
        
        acc = (
            x_pred[range(batch_size),masking_idx].argmax(-1) ==
            x[range(batch_size),masking_idx]
        ).float().mean()
        
        self.log(
            f'{step}_cross_entropy',
            xent,
            batch_size=batch_size,
            sync_dist=step=='val'
        )
        
        self.log(
            f'{step}_accuracy',
            acc,
            batch_size=batch_size,
            sync_dist=step=='val'
        )
        
        return xent

    def training_step(self, batch, batch_idx):
        return self.step(batch, step='train')
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, step='val')
        
#     def predict_step(self, batch, batch_idx=None):
#         return self.step(batch, step='predict')
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min')
        return {
           'optimizer': opt,
           'lr_scheduler': sched,
           'monitor': 'train_cross_entropy'
        }