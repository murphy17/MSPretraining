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
        max_length,
        dropout,
        output_weights,
        lr
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.model_depth = model_depth
        self.num_residues = num_residues
        self.max_length = max_length
        self.output_dim = output_dim
        self.dropout = dropout
        self.output_weights = output_weights
        self.lr = lr
        
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
        
        losses = []
        metrics = defaultdict(list)
        
        for k in range(self.output_dim):
            loss = F.binary_cross_entropy_with_logits(
                y_pred[:,k], y[:,k].float()
            )
            losses.append(loss * self.output_weights[k][1])
            
            auc = auroc(y_pred[:,k], y[:,k])
            metrics['auc'].append(auc)
            
            acc = accuracy(
                y_pred[:,k], y[:,k],
                num_classes=1, 
                average='macro'
            )
            metrics['acc'].append(acc)
            
        loss = torch.stack(losses).mean()
        
        if step != 'predict':
            self.log(f'{step}_loss',loss,batch_size=batch_size)

            for m in metrics:
                for i,(k,w) in enumerate(self.output_weights):
                    if w==0: continue
                    self.log(
                        f'{step}_{m}_{k}',
                        metrics[m][i],
                        batch_size=batch_size,
#                         sync_dist=step=='val'
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
        super().__init__(**kwargs,max_length=None)
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
            max_length=None,
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
    
# doens't work
# class RNNModel(SequenceModel):
#     def __init__(
#         self,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
        
#         self.encoder = nn.LSTM(
#             input_size=self.model_dim,
#             hidden_size=self.model_dim,
#             num_layers=self.model_depth,
#             batch_first=True,
#             dropout=self.dropout,
#             bidirectional=False
#         )
        
#         self.classifier = nn.Linear(
#             self.model_dim, 
#             self.output_dim
#         )
        
#     def forward(self, x):
#         x = self.embedding(x)
#         x, _ = self.encoder(x)
#         x = x[:,-1]
#         x = self.classifier(x)
#         return x
    
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

        self.transformer = MSTransformer.load_from_checkpoint(checkpoint)
        if naive:
            self.transformer = MSTransformer(**dict(self.transformer.hparams))
        self.transformer.requires_grad_(not fixed_weights)

#         self.classifier = nn.Sequential(
#             nn.Linear(self.model_dim, self.model_dim),
#             nn.LeakyReLU(0.2,inplace=True),
#             nn.Linear(self.model_dim, self.output_dim)
#         )
        
        self.classifier = ESMAttention1d(self.max_length, self.model_dim, self.output_dim)

    def forward(self, x, x_mask):
        x, _ = self.transformer.encoder(x, x_mask)
        x = self.classifier(x, x_mask)
        return x


from sequence_models.pretrained import load_model_and_alphabet
from sequence_models.structure import Attention1d

class ESMAttention1d(nn.Module):
    """Outputs of the ESM model with the attention1d"""
    def __init__(self, max_length, d_embedding, d_out):
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
        self.classifier = ESMAttention1d(self.max_length, self.model_dim, self.output_dim)

    def unbatch(self, batch):
        batch_size = len(batch['sequence'])
        x, = self.collater([[s] for s in batch['sequence']])
        x = x.to(self.device)
        x_mask = batch['x_mask'].unsqueeze(-1)
        y = batch['y']
        return (x, x_mask, y), batch_size
        
    def forward(self, x, x_mask):
        x = self.encoder(x, x_mask)
        x = self.classifier(x, x_mask)
        return x
