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
        
        self.residue_embedding = nn.Embedding(
            len(self.residues)+1, # CLS token 
            model_dim,
            padding_idx=0
        )
        
        self.charge_embedding = nn.Embedding(
            len(self.parent_charges), 
            model_dim
        )
        
        self.ce_embedding = nn.Sequential(
            nn.Linear(1, model_dim, bias=False)
        )

        self.positional_encoding = PositionalEncoding(
            d_model=model_dim,
            max_len=2*max_length, # striding
            dropout=dropout
        ).requires_grad_(False)

        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads, 
            num_encoder_layers=model_depth, 
            num_decoder_layers=model_depth,
            dim_feedforward=model_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Linear(model_dim, np.prod(self.output_dim))
        
        # self.dropout = nn.Dropout(p=dropout)
        
    def encoder(
        self,
        x_src,
        x_src_mask
    ):
        x_src = self.residue_embedding(x_src)
        x_src = self.positional_encoding(x_src, offset=0, stride=2)
        
        x_mem = self.transformer.encoder(
            src=x_src, 
            src_key_padding_mask=~x_src_mask
        )
        x_mem_mask = x_src_mask
        
        return x_mem, x_mem_mask
    
    def decoder(
        self,
        x_mem,
        x_mem_mask,
        charge,
        ce,
        softmax
    ):
        batch_size, max_residues, _ = x_mem.shape
        max_bonds = max_residues - 1
        
        x_tgt = torch.zeros_like(x_mem[:,1:])
        x_tgt = self.positional_encoding(x_tgt, offset=1, stride=2)
        x_tgt_mask = x_mem_mask[:,1:]
            
        charge = charge - min(self.parent_charges)
        charge = charge.view(-1,1).long()
        x_mem = x_mem + self.charge_embedding(charge)
        
        ce = ce.view(-1,1).float()
        x_mem = x_mem + self.ce_embedding(ce).unsqueeze(1)
        
        y = self.transformer.decoder(
            tgt=x_tgt, 
            memory=x_mem,
            tgt_key_padding_mask=~x_tgt_mask.bool(),
            memory_key_padding_mask=~x_mem_mask.bool()
        )
        y = self.classifier(y)
        
        if softmax:
            y = y.flatten(1)
            y = torch.softmax(y, dim=1)
            y = y.reshape(y.shape)
            
        y = y.reshape(-1, max_bonds, *self.output_dim)
        
        return y
        
    def forward(
        self, 
        sequence,
        sequence_mask,
        charge,
        ce,
        softmax
    ):
        x_mem, x_mem_mask = self.encoder(
            x_src=sequence.long(), 
            x_src_mask=sequence_mask.bool()
        )
        y = self.decoder(
            x_mem=x_mem,
            x_mem_mask=x_mem_mask,
            charge=charge.long(),
            ce=ce.float(),
            softmax=softmax
        )
        return y

    def masked_loss(self, loss_fn, input, target, mask):
        mask = mask.bool()
        batch_size = input.shape[0]
        loss = 0
        for input_i, target_i, mask_i in zip(input, target, mask):
            loss += loss_fn(input_i[mask_i].view(1,-1), target_i[mask_i].view(1,-1))
        loss /= batch_size
        return loss
    
    def step(self, batch, step):
        batch_size = batch['x'].shape[0]

        y = batch['y']
        y_mask = batch['y_mask']
        
        y_pred = self(
            sequence=batch['x'],
            sequence_mask=batch['x_mask'],
            charge=batch['charge'],
            ce=batch['collision_energy'],
            softmax=step=='predict'
        )
        
        y_total = y.flatten(1).sum(1).view(batch_size,1,1,1,1)
        
        if step=='predict':
            # renormalize to area of observed fragments
            y_pred /= (y_pred * (y > 0)).flatten(1).sum(1).view(batch_size,1,1,1,1)
            y_pred *= y_total
            return y_pred

        loss = self.masked_loss(
            F.cross_entropy, 
            y_pred, 
            y / y_total,
            y_mask
        )
        
        err = self.masked_loss(
            lambda a, b: ((torch.softmax(a,dim=1) * b.sum() - b) / (b+1)).abs().mean(), 
            y_pred.view(batch_size,-1),
            y.view(batch_size,-1), 
            y_mask.view(batch_size,-1)
        )
        
        if step != 'predict':
            self.log(
                f'{step}_cross_entropy',
                loss,
                batch_size=batch_size,
                sync_dist=step=='valid'
            )
            self.log(
                f'{step}_rel_abs_err',
                err,
                batch_size=batch_size,
                sync_dist=step=='valid'
            )
        
        return loss, err
    
    def training_step(self, batch, batch_idx):
        loss, err = self.step(batch, step='train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, step='valid')
        
    def predict_step(self, batch, batch_idx=None):
        return self.step(batch, step='predict')
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt