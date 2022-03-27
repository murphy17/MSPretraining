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
#         residues,
#         ions,
#         parent_min_charge,
#         parent_max_charge,
#         fragment_min_charge,
#         fragment_max_charge,
#         losses,
        model_dim,
        model_depth,
        num_heads,
        lr,
        dropout, 
        max_length,
        use_cls_token,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
#         self.residues = residues
#         self.ions = ions
#         self.parent_charges = range(parent_min_charge, parent_max_charge + 1)
#         self.fragment_charges = range(fragment_min_charge, fragment_max_charge + 1)
#         self.losses = losses

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
        self.use_cls_token = use_cls_token
        
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
        
    def _encode_src(self, sequence, sequence_mask):
        batch_size, max_residues = sequence.shape
        # prepend CLS token
        if self.use_cls_token:
            cls_token = len(self.residues) * torch.ones_like(sequence[:,[0]])
            x = torch.cat([cls_token,sequence],axis=1)
            x_mask = torch.cat([cls_token.bool(),sequence_mask],axis=1)
        else:
            x = sequence
            x_mask = sequence_mask
        x = self.residue_embedding(x)
        if self.use_cls_token:
            x[:,1:] = self.positional_encoding(x[:,1:], offset=0, stride=2)
        else:
            x = self.positional_encoding(x, offset=0, stride=2)
        return x, x_mask
    
    def _encode_tgt(self, sequence, fragment_mask):
        batch_size = sequence.shape[0]
        max_bonds = sequence.shape[1] - 1
        x = torch.zeros(batch_size,max_bonds,self.model_dim,device=self.device)
        x = self.positional_encoding(x, offset=1, stride=2)
        x_mask = fragment_mask
        return x, x_mask
    
    def _encode_mem(self, z, charge, ce):
        charge = charge - min(self.parent_charges)
        charge = charge.view(-1,1)
        z = z + self.charge_embedding(charge)
        ce = ce.view(-1,1)
        z = z + self.ce_embedding(ce).unsqueeze(1)
        return z
        
    def encoder(
        self,
        sequence,
        sequence_mask,
    ):
        batch_size, max_residues = sequence.shape
        max_bonds = max_residues - 1
        
        x_src, x_src_mask = self._encode_src(
            sequence.long(), 
            sequence_mask.bool()
        )
        
        z = self.transformer.encoder(
            src=x_src, 
            src_key_padding_mask=~x_src_mask
        )
        
        if self.use_cls_token:
            z = z[:,[0]]
        
        return z
    
    def decoder(
        self,
        z,
        sequence,
        charge,
        ce,
        fragment_mask,
        with_logits=False
    ):
        batch_size, max_residues = sequence.shape
        max_bonds = max_residues - 1
        
        x_tgt, x_tgt_mask = self._encode_tgt(
            sequence.long(),
            fragment_mask.bool()
        )
            
        z = self._encode_mem(z, charge.long(), ce.float())
        
        z = self.transformer.decoder(
            tgt=x_tgt, 
            memory=z,
            tgt_key_padding_mask=~x_tgt_mask.bool(),
            memory_key_padding_mask=None if self.use_cls_token else ~x_src_mask.bool()
        )
        
        y_pred = self.classifier(z)
        
        if not with_logits:
            y_pred = y_pred.flatten(1)
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.reshape(y_pred.shape)
            
        y_pred = y_pred.reshape(-1, max_bonds, *self.output_dim)
        
        return y_pred
        
    def forward(
        self, 
        sequence,
        charge,
        ce,
        sequence_mask, 
        fragment_mask,
        with_logits=False
    ):
        z = self.encoder(
            sequence, 
            sequence_mask
        )
        y_pred = self.decoder(
            z,
            sequence,
            charge,
            ce,
            fragment_mask,
            with_logits
        )
        return y_pred
#         batch_size, max_residues = sequence.shape
#         max_bonds = max_residues - 1
        
#         x_src, x_src_mask = self._encode_src(sequence, sequence_mask)
#         x_tgt, x_tgt_mask = self._encode_tgt(sequence, fragment_mask)
        
#         z = self.transformer.encoder(
#             src=x_src, 
#             src_key_padding_mask=~x_src_mask
#         )
#         z = self._encode_mem(z, charge, ce)
#         z = self.transformer.decoder(
#             tgt=x_tgt, 
#             memory=z,
#             tgt_key_padding_mask=~x_tgt_mask,
#             memory_key_padding_mask=None if self.use_cls_token else ~x_src_mask
#         )
        
#         y_pred = self.classifier(z)
        
#         if not with_logits:
#             y_pred = y_pred.flatten(1)
#             y_pred = torch.softmax(y_pred, dim=1)
#             y_pred = y_pred.reshape(y_pred.shape)
            
#         y_pred = y_pred.reshape(-1, max_bonds, *self.output_dim)
        
#         return y_pred

    def _masked_loss(self, loss_fn, input, target, mask):
        batch_size = input.shape[0]
        loss = 0
        for input_i, target_i, mask_i in zip(input, target, mask):
            loss += loss_fn(input_i[mask_i].view(1,-1), target_i[mask_i].view(1,-1))
        loss /= batch_size
        return loss
    
    def step(self, batch, predict_step=False):
        batch_size = batch['x'].shape[0]

        y = batch['y']
        y_mask = batch['y_mask']
        
        y_pred = self(
            sequence=batch['x'],
            charge=batch['charge'],
            ce=batch['collision_energy'],
            sequence_mask=batch['x_mask'],
            fragment_mask=batch['x_mask'][:,1:],
            with_logits=not predict_step
        )
        
        y_total = y.flatten(1).sum(1).view(batch_size,1,1,1,1)
        
        if predict_step:
            # renormalize to area of observed fragments
            y_pred /= (y_pred * (y > 0)).flatten(1).sum(1).view(batch_size,1,1,1,1)
            y_pred *= y_total
            return y_pred

        loss = self._masked_loss(
            F.cross_entropy, 
            y_pred, 
            y / y_total,
            y_mask
        )
        
        err = self._masked_loss(
            lambda a, b: ((torch.softmax(a,dim=1) * b.sum() - b) / (b+1)).abs().mean(), 
            y_pred.view(batch_size,-1),
            y.view(batch_size,-1), 
            y_mask.view(batch_size,-1)
        )
        
        return loss, err
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['x'].shape[0]
        loss, err = self.step(batch)
        self.log('train_cross_entropy',loss,batch_size=batch_size)
        self.log('train_rel_abs_err',err,batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = batch['x'].shape[0]
        loss, err = self.step(batch)
        self.log('valid_cross_entropy',loss,batch_size=batch_size,sync_dist=True)
        self.log('valid_rel_abs_err',err,batch_size=batch_size,sync_dist=True)
        
    def predict_step(self, batch, batch_idx=None):
        return self.step(batch, predict_step=True)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt