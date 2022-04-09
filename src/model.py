import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

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

class MSTransformer(pl.LightningModule):
    def __init__(
        self,
        model_dim,
        model_depth,
        num_heads,
        lr,
        dropout, 
        max_length,
        temperature,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
 
        self.residues = C.alphabet
        self.ions = C.ions
        self.parent_charges = range(C.min_charge, C.max_charge + 1)
        self.fragment_charges = range(C.min_frag_charge, C.max_frag_charge + 1)
        self.losses = C.losses
        
        self.model_dim = model_dim
        self.output_dim = (
            len(self.ions), 
            len(self.fragment_charges),
            len(self.losses)
        )
        self.max_length = max_length
        self.model_depth = model_depth
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.temperature = temperature

        self.positional_encoding = PositionalEncoding(
            d_model=model_dim,
            max_len=max_length,
            dropout=dropout
        ).requires_grad_(False)

        self.encoder = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads, 
            num_encoder_layers=model_depth, 
            num_decoder_layers=model_depth,
            dim_feedforward=model_dim,
            dropout=dropout,
            batch_first=True
        ).encoder
        
        self.embedding = nn.Linear(
            np.prod(self.output_dim), # blechchhh
            self.model_dim
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.model_dim, self.model_dim),
        )
        
    def forward(self, x, x_mask):
        x = x.flatten(2)
        x = x / (1+x.sum((1,2),keepdim=True))
        x_mask = x_mask.sum((2,3,4)) # sketchy; i don't like this! at all!
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(src=x, src_key_padding_mask=x_mask==0)
        x = x.mean(1)
        return x
    
    def simclr_loss(self, z1, z2, temperature):
        with torch.cuda.amp.autocast(enabled=False):
            z1 = z1.to(torch.float32)
            z2 = z2.to(torch.float32)
            
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            
            sim12 = z1 @ z2.T / temperature

            pos_loss = -torch.diag(sim12).mean()

            neg_loss = 0.5 * (
                torch.logsumexp(sim12, dim=0).mean() +
                torch.logsumexp(sim12, dim=1).mean()
            )
            
        return pos_loss + neg_loss

    def step(self, batch, predict_step=False):
        if predict_step:
            batch1 = batch2 = batch
        else:
            batch1, batch2 = batch
        
        batch_size = batch1['x'].shape[0]

        x1, x2 = batch1['y'], batch2['y']
        x1_mask, x2_mask = batch1['y_mask'], batch2['y_mask']
        
        z1 = self(x1, x1_mask)
        z2 = self(x2, x2_mask)
        
        if predict_step:
            return z1
        
        h1 = self.projection_head(z1)
        h2 = self.projection_head(z2)
        
        # simclr loss, pairing on sequence
        loss = self.simclr_loss(h1, h2, self.temperature)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        batch1, batch2 = batch
        batch_size = batch1['x'].shape[0]
        loss = self.step(batch)
        self.log('train_infonce_loss',loss,batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch1, batch2 = batch
        batch_size = batch1['x'].shape[0]
        loss = self.step(batch)
        self.log('valid_infonce_loss',loss,batch_size=batch_size,sync_dist=True)
        
    def predict_step(self, batch, batch_idx=None):
        return self.step(batch, predict_step=True)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt