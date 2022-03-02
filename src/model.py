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
            len(self.residues), 
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
        
#         self.lstm_encoder = nn.LSTM(
#             input_size=model_dim,
#             hidden_size=model_dim,
#             proj_size=model_dim//2,
#             num_layers=model_depth,
#             dropout=dropout,
#             batch_first=True,
#             bidirectional=True
#         )
#         self.lstm_decoder = nn.LSTM(
#             input_size=model_dim,
#             hidden_size=model_dim,
#             proj_size=model_dim//2,
#             num_layers=model_depth,
#             dropout=dropout,
#             batch_first=True,
#             bidirectional=True
#         )
        
        self.classifier = nn.Linear(model_dim, np.prod(self.output_dim))
        
    def _encode_src(self, sequence):
        x = self.residue_embedding(sequence)
        x = self.positional_encoding(x, offset=0, stride=2)
        return x
    
    def _encode_tgt(self, charge, ce, max_bonds):
        charge = charge - min(self.parent_charges)
        charge = charge.view(-1,1).expand(-1,max_bonds)
        ce = ce.view(-1,1).expand(-1,max_bonds)
        x = (
            self.charge_embedding(charge) +
            self.ce_embedding(ce.unsqueeze(-1))
        )
        x = self.positional_encoding(x, offset=1, stride=2)
        return x
        
    def forward(
        self, 
        sequence, 
        charge,
        ce,
        sequence_mask=None, 
        fragment_mask=None,
        with_logits=False
    ):
        batch_size, max_residues = sequence.shape
        max_bonds = max_residues - 1
        
        if sequence_mask is None:
            sequence_mask = torch.ones(batch_size, max_residues, device=self.device)
        x_src_mask = sequence_mask
        
        if fragment_mask is None:
            fragment_mask = torch.ones(batch_size, max_bonds, device=self.device)
        x_tgt_mask = fragment_mask
        
        x_src = self._encode_src(sequence)
        x_src *= x_src_mask.unsqueeze(-1) # unsure
        
        x_tgt = self._encode_tgt(charge, ce, max_bonds)
        x_tgt *= x_tgt_mask.unsqueeze(-1) # unsure
        
        z = self.transformer(
            src = x_src,
            tgt = x_tgt,
            src_key_padding_mask = ~x_src_mask,
            memory_key_padding_mask = ~x_src_mask,
            tgt_key_padding_mask = ~x_tgt_mask
        )
        y_pred = self.classifier(z)
        
#         z, _ = self.lstm_encoder(x_src * x_src_mask.unsqueeze(-1))
#         z = z[:,:max_bonds] + x_tgt
#         z, _ = self.lstm_decoder(z * x_tgt_mask.unsqueeze(-1))
#         y_pred = self.classifier(z)
        
        if not with_logits:
            y_pred = y_pred.flatten(1)
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.reshape(y_pred.shape)
            
        y_pred = y_pred.reshape(-1, max_bonds, *self.output_dim)
        
        return y_pred

    def _masked_loss(self, loss_fn, input, target, mask):
        batch_size = input.shape[0]
        loss = 0
        for input_i, target_i, mask_i in zip(input, target, mask):
            loss += loss_fn(input_i[mask_i].view(1,-1), target_i[mask_i].view(1,-1))
        loss /= batch_size
        return loss
    
    def step(self, batch, predict_step=False):
        batch_size = batch['x'].shape[0]

        y = batch['y'].float()
        y_mask = batch['y_mask'].bool()
        
        y_pred = self(
            sequence=batch['x'].long(),
            charge=batch['charge'].long(),
            ce=batch['collision_energy'].float(),
            sequence_mask=batch['x_mask'].bool(),
            fragment_mask=batch['x_mask'].bool()[:,1:],
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