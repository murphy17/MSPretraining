import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

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
        residues,
        ions,
        parent_min_charge,
        parent_max_charge,
        fragment_min_charge,
        fragment_max_charge,
        losses,
        model_dim,
        model_depth,
        num_heads,
        lr,
        dropout, 
        max_length
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.residues = residues
        self.ions = ions
        self.parent_charges = range(parent_min_charge, parent_max_charge + 1)
        self.fragment_charges = range(fragment_min_charge, fragment_max_charge + 1)
        self.losses = losses
        
        self.model_dim = model_dim
        self.output_dim = (len(self.ions), len(self.fragment_charges), len(self.losses))
        self.max_length = max_length
        self.model_depth = model_depth
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        
        self.residue_embedding = nn.Embedding(len(self.residues), model_dim)
        self.charge_embedding = nn.Embedding(len(self.parent_charges), model_dim)
        self.ce_embedding = nn.Linear(1, model_dim)

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
        
    def forward(
        self, 
        sequence, 
        charge,
        ce,
        sequence_mask=None, 
        fragment_mask=None,
        with_logits=False
    ):
        batch_size, max_residues = sequence.shape[:2]
        max_bonds = max_residues - 1
        
        if sequence_mask is None:
            sequence_mask = torch.ones(batch_size, max_residues, device=self.device)
        x_src_mask = sequence_mask
        
        if fragment_mask is None:
            fragment_mask = torch.ones(batch_size, max_bonds, device=self.device)
        x_tgt_mask = fragment_mask
        
        x_src = self.residue_embedding(sequence)
        x_src = self.positional_encoding(x_src, offset=0, stride=2)
        x_src *= x_src_mask.unsqueeze(-1) # unsure
        
        x_tgt = self.charge_embedding(charge.unsqueeze(-1))
        x_tgt += self.ce_embedding(ce.unsqueeze(-1)).unsqueeze(1)
        x_tgt = x_tgt.expand(-1,max_bonds,-1)
        x_tgt = self.positional_encoding(x_tgt, offset=1, stride=2)
        x_tgt *= x_tgt_mask.unsqueeze(-1) # unsure
            
        y_pred = self.transformer(
            src=x_src,
            tgt=x_tgt,
            src_key_padding_mask=x_src_mask != 1,
            memory_key_padding_mask=x_src_mask != 1,
            tgt_key_padding_mask=x_tgt_mask != 1
        )
        
        y_pred = self.classifier(y_pred)
        
        if not with_logits:
            y_pred = torch.softmax(y_pred.flatten(1), dim=1).reshape(y_pred.shape)
            
        y_pred = y_pred.reshape(-1, max_bonds, *self.output_dim)
        
        return y_pred

    def masked_loss(self, loss_fn, input, target, mask):
        batch_size = input.shape[0]
        loss = 0
        for input_i, target_i, mask_i in zip(input, target, mask):
            loss += loss_fn(input_i[mask_i].view(1,-1), target_i[mask_i].view(1,-1))
        loss /= batch_size
        return loss
    
    def step(self, batch, predict_step=False):
        batch_size, max_residues = batch['x'].shape[:2]
        max_bonds = max_residues - 1

        y = batch['y'].float()
        y_mask = batch['y_mask'].bool()
        
        y_pred = self(
            sequence=batch['x'].long(),
            charge=batch['charge'].long() - self.parent_charges[0],
            ce=batch['collision_energy'].float(),
            sequence_mask=batch['x_mask'].bool(),
            fragment_mask=batch['x_mask'][:,1:].bool()
        )
        
        if predict_step:
            # renormalize to area of observed fragments
            y_pred /= (y_pred * (y > 0)).flatten(1).sum(1)
            y_pred *= y.flatten(1).sum(1)
            return y_pred

        y_total = y.flatten(1).sum(1).view(batch_size,1,1,1,1)
        
        loss = self.masked_loss(F.cross_entropy, y_pred, y / y_total, y_mask)
        
        err = self.masked_loss(
            lambda a, b: ((torch.softmax(a,dim=1) * b.sum() - b) / (b+1)).abs().mean(), 
            y_pred.view(batch_size,-1),
            y.view(batch_size,-1), 
            y_mask.view(batch_size,-1)
        )
        
        return loss, err
    
    def training_step(self, batch, batch_idx):
        loss, err = self.step(batch)
        self.log('train_cross_entropy',loss)
        self.log('train_rel_abs_err',err)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, err = self.step(batch)
        self.log('valid_cross_entropy',loss)
        self.log('valid_rel_abs_err',err)
        
    def predict_step(self, batch, batch_idx=None):
        return self.step(batch, predict_step=True)
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt