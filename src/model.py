import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from sequence_models.convolutional import ByteNetLM, MaskedConv1d

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
    
class MSDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        output_dim
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.conv1 = MaskedConv1d(
            in_channels=input_dim,
            out_channels=model_dim,
            kernel_size=2 # residues -> bonds
        )
        self.conv1.padding = 0 # residues -> bonds
        self.relu1 = nn.ReLU()
        self.conv2 = MaskedConv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=1
        )
        self.relu2 = nn.ReLU()
        self.conv3 = MaskedConv1d(
            in_channels=model_dim,
            out_channels=output_dim,
            kernel_size=1
        )
    
    def forward(self, x, input_mask):
        x = self.conv1(x, input_mask)
        x = self.relu1(x)
        x = self.conv2(x, input_mask[:,1:])
        x = self.relu2(x)
        x = self.conv3(x, input_mask[:,1:])
        return x
    
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
        
        self.aa_lambda = 0.01

        self.encoder = ByteNetLM(
            n_tokens=self.input_dim,
            d_embedding=8,
            d_model=self.model_dim,
            n_layers=self.model_depth,
            kernel_size=5,
            r=self.model_dim,
            padding_idx=0, 
            causal=False,
            dropout=self.dropout,
            activation='gelu'
        )
        self.encoder.decoder = nn.Identity()
        # alternative would be one of these per charge x energy
        self.decoder = MSDecoder(
            input_dim=self.model_dim+self.condition_dim,
            model_dim=self.model_dim,
            output_dim=np.prod(self.output_dim)
        )
#         self.decoder = ByteNetLM(
#             n_tokens=np.prod(self.output_dim),
#             d_embedding=self.model_dim+self.condition_dim, 
#             **self.bytenet_params
#         )
#         self.decoder.embedder.embedder = nn.Identity()

        self.classifier = nn.Linear(
            self.model_dim, 
            self.input_dim,
        )

    def forward(self, x, x_mask, c, *, softmax=False, embedding=False):
        z = self.encoder(x, input_mask=x_mask.unsqueeze(-1))
        # x = self.dropout(x)
        # condition on charge and energy
        x = torch.cat([z,c.unsqueeze(1).expand(-1,x.shape[1],-1)],dim=-1)
        x = self.decoder(x, input_mask=x_mask.unsqueeze(-1))
        # residues -> bonds
        # x = x[:,:-1] * x_mask[:,1:].unsqueeze(-1)
        if softmax:
            x = torch.softmax(x.flatten(1), dim=1).reshape(x.shape)
        if embedding:
            return x, z
        return x

    def cross_entropy(self, logits, target, mask):
        batch_size = logits.shape[0]
        target = target.flatten(1)
        logits = logits.flatten(1)
        mask = mask.flatten(1)
        xent = -((target * F.log_softmax(logits,1)) * mask).sum(1).mean()
        return xent
    
    def r_squared(self, y_pred, y, mask):
        batch_size = y.shape[0]
        y_pred = y_pred.flatten(1)
        y = y.flatten(1)
        mask = mask.flatten(1)
        y_mean = (y * mask).sum(1) / mask.sum(1)
        ss_res = ((y - y_pred).square() * mask).sum(1)
        ss_tot = ((y - y_mean.unsqueeze(1)).square() * mask).sum(1)
        r2 = (1 - ss_res / ss_tot).mean()
        return r2
    
    def step(self, batch, step):
        batch_size = batch['x'].shape[0]
        max_length = batch['x'].shape[1]

        x = batch['x']
        x_mask = batch['x_mask']
        
        c = torch.stack([
            batch['charge'],
            batch['collision_energy']
        ],-1).float()
        
        y = batch['y']
        # y_mask = batch['y_mask']
        y_mask = x_mask[:,1:].view(batch_size,max_length-1,1,1,1).expand_as(batch['y_mask'])
        
        y_pred, z = self(x, x_mask, c, softmax=False, embedding=True)
        y_pred = y_pred.reshape(*y_pred.shape[:-1],*self.output_dim)
        
        y_total = y.flatten(1).sum(1).view(batch_size,1,1,1,1)
        
        xent_ms = self.cross_entropy(y_pred, y / y_total, y_mask)
        
        y_pred[y_mask==0] = -float('inf')
        y_pred = torch.softmax(y_pred.flatten(1),1).reshape(y_pred.shape)
        y_pred *= y_total
        
        rsqr = self.r_squared(y_pred, y, y_mask)
        
        # regularize the embedding: make the AA predictable
        # ... from the local embedding only? from the rest?
        # hmm... ok...
        # what if you set this up very very differently
        # "given the spectrum and the punctured sequence, predict the AA"
        
        x_pred = self.classifier(z)
        xent_aa = F.cross_entropy(
            x_pred.reshape(-1,self.input_dim),
            x.flatten()
        )
        
        # acc_aa = (x_pred.argmax(-1) == x).float().mean()
        
        loss = xent_ms + self.aa_lambda * xent_aa
        
        if step == 'predict':
            return y_pred
        else:
            self.log(
                f'{step}_cross_entropy',
                xent_ms,
                batch_size=batch_size,
                sync_dist=step=='val'
            )
            self.log(
                f'{step}_aa_cross_entropy',
                xent_aa,
                batch_size=batch_size,
                sync_dist=step=='val'
            )
            self.log(
                f'{step}_r_squared',
                rsqr,
                batch_size=batch_size,
                sync_dist=step=='val'
            )
            return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, step='train')
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, step='val')
        
    def predict_step(self, batch, batch_idx=None):
        return self.step(batch, step='predict')
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt