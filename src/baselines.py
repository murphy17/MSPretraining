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
        balance_classes,
        output_weights,
        lr
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.model_depth = model_depth
        self.num_residues = num_residues
        self.output_dim = output_dim
        self.dropout = dropout
        self.balance_classes = balance_classes
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
            if self.balance_classes:
                pos_weight = (1+(y[:,k]==0).sum()) / (1+(y[:,k]==1).sum())
            else:
                pos_weight = None
            
            loss = F.binary_cross_entropy_with_logits(
                y_pred[:,k], y[:,k].float(), 
                pos_weight=pos_weight
            )
            losses.append(loss * self.output_weights[k])
            
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
                for k,w in enumerate(self.output_weights):
                    if w==0: continue
                    self.log(f'{step}_{m}_{k}',metrics[m][k],batch_size=batch_size)
            
        return loss, metrics, y_pred
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx, step='train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, step='valid')
            
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
    
from sequence_models.pretrained import load_model_and_alphabet

class CARPModel(SequenceModel):
    def __init__(
        self,
        fixed_weights,
        **kwargs
    ):
        super().__init__(
            model_dim=30,
            model_depth=None,
            num_residues=None,
            dropout=None,
            **kwargs
        )

        self.encoder, self.collater = load_model_and_alphabet('carp_600k')
        self.encoder.requires_grad_(not fixed_weights)

        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm1d(self.model_dim),
            nn.Linear(self.model_dim, self.output_dim)
        )

    def unbatch(self, batch):
        batch_size = len(batch['sequence'])
        x, = self.collater([[s] for s in batch['sequence']])
        x = x.to(self.device)
        x_mask = batch['x_mask']
        y = batch['y']
        return (x, x_mask, y), batch_size
        
    def forward(self, x, x_mask):
        x = self.encoder(x)
        x = x * x_mask.unsqueeze(-1)
        x = x.sum(1) / x_mask.sum(-1).view(-1,1)
        x = self.classifier(x)
        return x

    