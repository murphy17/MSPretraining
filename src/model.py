import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from sequence_models.convolutional import ByteNet, ByteNetLM, MaskedConv1d

from .constants import MSConstants
C = MSConstants()

# misnomer
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
        
        self.encoder_dropout = 0
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
            dropout=self.encoder_dropout,
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
            dropout=self.encoder_dropout,
            activation='gelu'
        )
        self.y_encoder.embedder = nn.Identity()
        
        # final AA classifier should be VERY SIMPLE!
        self.conv1 = MaskedConv1d(
            self.model_dim * 2, 
            self.model_dim,
            kernel_size=1
        )
        self.relu = nn.ReLU() # GELU?
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
        if self.training and self.dropout > 0:
            y_mask = torch.rand_like(y) < self.dropout
            y[y_mask] = 0
        
        # normalize spectrum; for domain reasons, but also fixes dropout ^^ ?
        y = y / y.flatten(1).sum(-1).clamp(1,float('inf')).view(-1,1,1,1,1)
        
        y_pad = torch.zeros(batch_size,1,*self.output_dim,device=self.device)
        y = torch.cat([y,y_pad],1)

        # to start, just a single AA
        masking_idx = torch.multinomial(padding_mask.float(), 1).squeeze()
        x_mask = torch.zeros_like(padding_mask,dtype=torch.bool)
        x_mask[range(batch_size),masking_idx] = 1
        x_masked = x.clone()
        x_masked[x_mask] = self.masking_idx
        
        # how well does it do with just sequence?
        if self.dropout == 1:
            y = torch.zeros_like(y)
            c = torch.zeros_like(c)

        x_pred = self(x_masked, y, c, padding_mask)
        
        # kludge
        if step == 'predict':
            return x_pred.argmax(-1), masking_idx

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
        
    def predict_step(self, batch, batch_idx=None):
        return self.step(batch, step='predict')
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt