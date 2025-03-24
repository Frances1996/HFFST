
import os 
import inspect
import importlib
import wget
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models
from .module import Spot_Encoder, Base_Encoder, CrossFusion, ImageEncoder_SWIN_TRANS, load_model_weights
from torchvision.models import DenseNet121_Weights
from torch.optim.lr_scheduler import StepLR






class HFFST(pl.LightningModule):
    def __init__(self, 
            num_genes=250,
            emb_dim=512,
            depth_t=2,
            depth_n=2,
            depth_s=2,
            depth_f=2,
            num_heads_t=8,
            num_heads_n=8,
            num_heads_s=8,
            num_heads_f=8,
            mlp_ratio_t=2.0,
            mlp_ratio_n=2.0,
            mlp_ratio_s=2.0,
            mlp_ratio_f=2.0,
            dropout_t=0.1,
            dropout_n=0.1,
            dropout_s=0.1,
            dropout_f=0.1,
            kernel_size=3,
            res_neighbor=(5, 5),
            learning_rate= 0.0001,
            backbone='resnet34',
            backbone_weights=r'weights\resnet34-b627a593.pth',
            ):
        super().__init__()
        
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        # Initialize best metrics
        self.best_loss = np.inf
        self.best_cor = -1
        
        self.num_genes = num_genes
        self.alpha = 0.3
        self.num_n = res_neighbor[0]
        self.backbone = backbone
        self.backbone_weights = backbone_weights

        # ***************************Backbone*******************************
        self.is_swintransfermer = False

        # Fine Encoder
        self.fine_tokenizer, conv1x1 = self.load_backbone(self.backbone, self.backbone_weights)
        if conv1x1 is not None:
            self.fine_tokenizer = nn.Sequential(*self.fine_tokenizer, conv1x1)
        self.fc_f = nn.Linear(emb_dim, num_genes)
        self.fine_encoder = Base_Encoder(emb_dim, depth_t, num_heads_t, int(emb_dim*mlp_ratio_t), dropout=dropout_t, resolution=(7,7))

        # Coarse Encoder
        self.coarse_encoder = Base_Encoder(emb_dim, depth_n, num_heads_n, int(emb_dim*mlp_ratio_n), dropout=dropout_n, resolution=(5,5))
        self.fc_c = nn.Linear(emb_dim, num_genes)

        # Spot Encoder
        self.medium_encoder = Spot_Encoder(emb_dim, depth_s, num_heads_s, int(emb_dim*mlp_ratio_s), dropout_s, kernel_size)
        self.fc_m = nn.Linear(emb_dim, num_genes)
    
        # Fusion Layer
        self.fusion1 = CrossFusion(emb_dim, depth_f, num_heads_f, int(emb_dim*mlp_ratio_f), dropout_f)
        self.fusion2 = CrossFusion(emb_dim, depth_f, num_heads_f, int(emb_dim*mlp_ratio_f), dropout_f)
        self.fusion3 = CrossFusion(emb_dim, depth_f, num_heads_f, int(emb_dim*mlp_ratio_f), dropout_f)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.9)


    def load_backbone(self, backbone, weights_path):
        """ Load the backbone model and return the feature extractor + optional conv1x1 layer """
        if backbone == "resnet34":
            model = load_model_weights("resnet34", weights_path)
            feature_extractor = nn.Sequential(*list(model.children())[:-2])
            output_channels = 512
        elif backbone == "resnet18":
            model = load_model_weights("resnet18", weights_path)
            feature_extractor = nn.Sequential(*list(model.children())[:-2])
            output_channels = 512
        elif backbone == "densenet121":
            model = load_model_weights("densenet121", weights_path)
            feature_extractor = list(model.children())[0]
            output_channels = model.features.norm5.num_features
        elif backbone == "swin_transformer":
            self.is_swintransfermer = True

            model = ImageEncoder_SWIN_TRANS()
            feature_extractor = nn.Sequential(*list(model.model.children()))
            output_channels = 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        if output_channels != 512:
            conv1x1 = nn.Conv2d(in_channels=output_channels, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            conv1x1 = None
        return feature_extractor, conv1x1

    def load_model(self):
        name = self.hparams.MODEL.name
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)


    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.MODEL.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.MODEL, arg)
        args1.update(other_args)
        return Model(**args1)


    def forward(self, x, x_total, position, neighbor, mask, pid=None, sid=None):
        x = x.cuda()
        mask = mask.cuda()
        
        # Target tokens
        fine_token = self.fine_tokenizer(x) # B x 512 x 7 x 7

        if self.is_swintransfermer == True:
            B, N, C = fine_token.shape  # (Batch, Token数, Channel)
            H = W = int(N ** 0.5)  # 计算 7×7
            fine_token = fine_token.view(B, H, W, C).permute(0, 3, 1, 2)
            fine_token = self.conv1x1(fine_token)

        _, dim, w, h = fine_token.shape
        fine_token = rearrange(fine_token, 'b d h w -> b (h w) d', d = dim, w=w, h=h)
        fine_token = self.fine_encoder(fine_token) # B x 49 x 512

        # spot tokens
        if pid == None:
            medium_token = self.medium_encoder(x_total, position.squeeze()).squeeze()  # N x 512
            if sid != None:
                medium_token = medium_token[sid]
        else:
            pid = pid.view(-1).cuda()
            sid = sid.view(-1).cuda()
            medium_token = torch.zeros((len(x_total), x_total[0].shape[1])).to(x.device)
            
            pid_unique = pid.unique()
            for pu in pid_unique:
                ind = int(torch.argmax((pid == pu).int()))
                x_g = x_total[ind].unsqueeze(0) # 1 x N x 512
                pos = position[ind]
                
                emb = self.medium_encoder(x_g.cuda(), pos.cuda()).squeeze()
                medium_token[pid == pu] = emb[sid[pid == pu]].float()

        # Coarse tokens
        coarse_token = self.coarse_encoder(neighbor.cuda(), mask) # B x 25 x 512

        # fusion tokens
        fusion1_token = self.fusion1(medium_token.unsqueeze(1), coarse_token, mask=mask)# (128, 512)
        fusion2_token = self.fusion2(medium_token.unsqueeze(1), fine_token, mask=mask)# (128, 512)
        fusion3_token = self.fusion3(fusion1_token, fusion2_token)# (128, 512)

        output_coarse = self.fc_c(coarse_token.mean(1)) # B x num_genes
        output_medium = self.fc_m(fusion1_token.squeeze()) # B x num_genes
        output = self.fc_f(fusion3_token.squeeze()) # B x num_genes

        return output, output_medium, output_coarse
    
    def training_step(self, batch, batch_idx):
        patch, exp, pid, sid, wsi, position, neighbor, mask = batch
        output, output_medium, output_coarse = self(patch, wsi, position, neighbor, mask, pid, sid)
        exp = exp.cuda()
        # Fusion loss
        loss = F.mse_loss(output.view_as(exp), exp)
        
        # Medium loss
        loss += F.mse_loss(output_medium.view_as(exp), exp) * (1-self.alpha)

        # Coarse loss
        loss += F.mse_loss(output_coarse.view_as(exp), exp) * (1-self.alpha)

        return loss
    
    
    
    def validation_step(self, batch, batch_idx):
        patch, exp, _, wsi, position, name, neighbor, mask = batch
        wsi = wsi.cuda()
        position = position.cuda()
        patch, exp, neighbor, mask = patch.squeeze().cuda(), exp.squeeze().cuda(), neighbor.squeeze().cuda(), mask.squeeze().cuda()
        pred, _, _ = self(patch, wsi, position, neighbor, mask)
        loss = F.mse_loss(pred.view_as(exp), exp)

        pred=pred.cpu().numpy().T
        exp=exp.cpu().numpy().T
        r=[]

        pred = pred.squeeze()
        for g in range(self.num_genes):
            r.append(pearsonr(pred[g], exp[g])[0])
        rr = torch.Tensor(r)
        
        self.get_meta(name)
        return {"val_loss":loss, "corr":rr}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        
        avg_corr = torch.stack(
            [x["corr"] for x in outputs])
        
        if self.best_cor < avg_corr.mean():

            self.best_cor = avg_corr.mean()
            self.best_loss = avg_loss
        
        self.log('valid_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('R', avg_corr.nanmean(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        patch, exp, sid, wsi, position, name, neighbor, mask = batch
        patch, exp, sid, neighbor, mask = patch.squeeze().cuda(), exp.squeeze().cuda(), sid.squeeze(), neighbor.squeeze().cuda(), mask.squeeze().cuda()
        
        if '10x_breast' in name[0]:
            wsi = wsi[0].unsqueeze(0)
            position = position[0]
            
            patches = patch.split(512, dim=0)
            neighbors = neighbor.split(512, dim=0)
            masks = mask.split(512, dim=0)
            sids = sid.split(512, dim=0)
            
            pred  = []
            for patch, neighbor, mask, sid in zip(patches, neighbors, masks, sids):
                outputs = self(patch, wsi, position, neighbor, mask, sid=sid, return_emb=True)
                p = outputs[0]
                
                pred.append(p)
                
            pred = torch.cat(pred, axis=0)
            
            ind_match = np.load(rf'data\test\{name[0]}\ind_match.npy', allow_pickle=True)
            self.num_genes = len(ind_match)
            pred = pred[:, ind_match]
            
        else:
            outputs = self(patch, wsi, position, neighbor.squeeze(), mask.squeeze())
            pred = outputs[0]
            
        mse = F.mse_loss(pred.view_as(exp), exp)
        mae = F.l1_loss(pred.view_as(exp), exp)
        
        pred=pred.cpu().numpy().T
        exp=exp.cpu().numpy().T
        
        r=[]
        for g in range(self.num_genes):
            r.append(pearsonr(pred[g], exp[g])[0])
        rr = torch.Tensor(r)
        
        self.get_meta(name)
        
        os.makedirs(f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}", exist_ok=True)
        np.save(f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/{name[0]}", pred.T)
        
        return {"MSE":mse, "MAE":mae, "corr":rr}

    def test_epoch_end(self, outputs):
        avg_mse = torch.stack(
            [x["MSE"] for x in outputs]).nanmean()

        avg_mae = torch.stack(
            [x["MAE"] for x in outputs]).nanmean()

        avg_corr = torch.stack(
            [x["corr"] for x in outputs]).nanmean(0)

        os.makedirs(f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}", exist_ok=True)
        torch.save(avg_mse.cpu(), f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/MSE")
        torch.save(avg_mae.cpu(), f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/MAE")
        torch.save(avg_corr.cpu(), f"final/{self.__class__.__name__}_{self.num_n}/{self.data}/{self.patient}/cor")

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def get_meta(self, name):
        if '10x_breast' in name[0]:
            self.patient = name[0]
            self.data = "test"
        else:
            name = name[0]
            self.data = name.split("+")[1]
            self.patient = name.split("+")[0]
            
            if self.data == 'her2st':
                self.patient = self.patient[0]
            elif self.data == 'stnet':
                self.data = "stnet"
                patient = self.patient.split('_')[0]
                if patient in ['BC23277', 'BC23287', 'BC23508']:
                    self.patient = 'BC1'
                elif patient in ['BC24105', 'BC24220', 'BC24223']:
                    self.patient = 'BC2'
                elif patient in ['BC23803', 'BC23377', 'BC23895']:
                    self.patient = 'BC3'
                elif patient in ['BC23272', 'BC23288', 'BC23903']:
                    self.patient = 'BC4'
                elif patient in ['BC23270', 'BC23268', 'BC23567']:
                    self.patient = 'BC5'
                elif patient in ['BC23269', 'BC23810', 'BC23901']:
                    self.patient = 'BC6'
                elif patient in ['BC23209', 'BC23450', 'BC23506']:
                    self.patient = 'BC7'
                elif patient in ['BC23944', 'BC24044']:
                    self.patient = 'BC8'
            elif self.data == 'skin':
                self.patient = self.patient.split('_')[0]

    def load_model(self):
        name = self.hparams.MODEL.name
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)


    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.MODEL.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.MODEL, arg)
        args1.update(other_args)
        return Model(**args1)
    
class CustomWriter(BasePredictionWriter):
    def __init__(self, pred_dir, write_interval, emb_dir=None, names=None):
        super().__init__(write_interval)
        self.pred_dir = pred_dir
        self.emb_dir = emb_dir
        self.names = names

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        for i, batch in enumerate(batch_indices[0]):
            torch.save(predictions[0][i][0], os.path.join(self.pred_dir, f"{self.names[i]}.pt"))
            torch.save(predictions[0][i][1], os.path.join(self.emb_dir, f"{self.names[i]}.pt"))
