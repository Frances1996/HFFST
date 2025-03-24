import itertools
from collections import OrderedDict
import torch
from torch import nn
from einops import rearrange
from timm.models.layers import to_2tuple
import timm
import torchvision



class PreNorm(nn.Module):
    def __init__(self, emb_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        if 'x_kv' in kwargs.keys():
            kwargs['x_kv'] = self.norm(kwargs['x_kv'])
        
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads = 4, dropout = 0., attn_bias=False, resolution=(5, 5)):
        super().__init__()
        
        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'
        
        dim_head = emb_dim // heads 
        project_out = not (heads == 1) 

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        
        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3, bias = False) 

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.attn_bias = attn_bias
        if attn_bias:
            points = list(itertools.product(
                range(resolution[0]), range(resolution[1])))
            N = len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])
            self.attention_biases = torch.nn.Parameter(
                torch.zeros(heads, len(attention_offsets)))
            self.register_buffer('attention_bias_idxs',
                                torch.LongTensor(idxs).view(N, N),
                                persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        if self.attn_bias:
            super().train(mode)
            if mode and hasattr(self, 'ab'):
                del self.ab
            else:
                self.ab = self.attention_biases[:, self.attention_bias_idxs]
        
    def forward(self, x, mask = None, return_attn=False):

        qkv = self.to_qkv(x).chunk(3, dim = -1) 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) 

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
        if self.attn_bias:
            qk += (self.attention_biases[:, self.attention_bias_idxs]
            if self.training else self.ab)
        
        if mask is not None:
            fill_value = torch.finfo(torch.float16).min
            ind_mask = mask.shape[-1]
            qk[:,:,-ind_mask:,-ind_mask:] = qk[:,:,-ind_mask:,-ind_mask:].masked_fill(mask==0, fill_value)

        attn_weights = self.attend(qk) # b h n n
        if return_attn:
            attn_weights_averaged = attn_weights.mean(dim=1)
        
        out = torch.matmul(attn_weights, v) 
        out = rearrange(out, 'b h n d -> b n (h d)')
    
        if return_attn:
            return self.to_out(out), attn_weights_averaged[:,0]
        else:
            return self.to_out(out)
        

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_dim, heads = 4, dropout = 0., attn_bias=False):
        super().__init__()
        
        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'
        
        dim_head = emb_dim // heads 
        project_out = not (heads == 1) 

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        
        self.to_q = nn.Linear(emb_dim, emb_dim, bias = False) 
        self.to_kv = nn.Linear(emb_dim, emb_dim * 2, bias = False) 

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, x_q, x_kv, mask=None, return_attn=False):

        q = self.to_q(x_q)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        kv = self.to_kv(x_kv).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv) 

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            fill_value = torch.finfo(torch.float16).min
            ind_mask = mask.shape[-1]
            qk[:,:,-ind_mask:,-ind_mask:] = qk[:,:,-ind_mask:,-ind_mask:].masked_fill(mask==0, fill_value)

        attn_weights = self.attend(qk) # b h n n
        if return_attn:
            attn_weights_averaged = attn_weights.mean(dim=1)
        
        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
    
        if return_attn:
            return self.to_out(out), attn_weights_averaged[:, 0]
        else:
            return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., attn_bias=False, resolution=(5,5)):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadAttention(emb_dim, heads = heads, dropout = dropout, attn_bias=attn_bias, resolution=resolution)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask=None, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x, mask=mask, return_attn=return_attn)
                x += attn_out # residual connection after attention      
                x = ff(x) + x # residual connection after feed forward net
                
            else:
                x = attn(x, mask=mask) + x # residual connection after attention      
                x = ff(x) + x # residual connection after feed forward net
            
        if return_attn:
            return x, attn_weights
        else:
            return x


class CrossEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0., attn_bias=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadCrossAttention(emb_dim, heads = heads, dropout = dropout, attn_bias=attn_bias)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x_q, x_kv, mask=None, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x_q, x_kv=x_kv, mask=mask, return_attn=return_attn)
                x_q += attn_out # residual connection after attention      
                x_q = ff(x_q) + x_q # residual connection after feed forward net
            else:
                x_q = attn(x_q, x_kv=x_kv, mask=mask) + x_q
                x_q = ff(x_q) + x_q # residual connection after feed forward net

        if return_attn:
            return x_q, attn_weights
        else:
            return x_q




class PEGH(nn.Module):
    def __init__(self, dim=512, kernel_size=None):
        super(PEGH, self).__init__()
        kernel1 = 3
        self.proj1 = nn.Conv2d(dim, dim, kernel1, padding=kernel1//2, bias=True, groups=dim)
        kernel2 = 5
        self.proj2 = nn.Conv2d(dim, dim, kernel2, padding=kernel2//2, bias=True, groups=dim)
        kernel3 = 7
        self.proj3 = nn.Conv2d(dim, dim, kernel3, padding=kernel3//2, bias=True, groups=dim)
        
    def forward(self, x, pos):

        pos = pos - pos.min(0)[0]
        x_sparse = torch.sparse_coo_tensor(pos.T , x.squeeze())# 按照postion的位置将x排列好,pos为所有spot的坐标，x为所有spot的feature
        x_dense = x_sparse.to_dense().permute(2,1,0).unsqueeze(dim=0)
        
        x_0 = self.proj1(x_dense)
        x_1 = self.proj2(x_dense)
        x_2 = self.proj3(x_dense)

        x_pos = x_0 + x_1 + x_2

        mask = (x_dense.sum(dim=1) != 0.)
        x_pos = x_pos.masked_fill(~mask, 0.) + x_dense
        x_pos_sparse = x_pos.squeeze().permute(2,1,0).to_sparse(2)
        x_out = x_pos_sparse.values().unsqueeze(dim=0)
        
        return x_out





class Spot_Encoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0., kernel_size=3):
        super().__init__()      
        
        self.pos_layer = PEGH(dim=emb_dim, kernel_size=kernel_size)
        
        self.layer1 = TransformerEncoder(emb_dim, 1, heads, mlp_dim, dropout)
        self.layer2 = TransformerEncoder(emb_dim, depth-1, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def foward_features(self, x, pos):
        # Translayer x1
        x = self.layer1(x) #[B, N, 384]

        # PEGH
        x = self.pos_layer(x, pos) #[B, N, 384]

        # Translayer x (depth-1)
        x = self.layer2(x) #[B, N, 384]
        x = self.norm(x)

        return x

    def forward(self, x, position):    
        x = self.foward_features(x, position) # 1 x N x 384

        return x



class Base_Encoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0., resolution=(5,5)):
        super().__init__()      
        
        self.layer = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout, attn_bias=True, resolution=resolution)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        
        if mask != None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            
        # Translayer
        x = self.layer(x, mask=mask) #[B, N, 512]
        x = self.norm(x)
        
        return x


class FusionEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()      
                
        self.cross_attention = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x_t=None, x_n=None, x_g=None, mask=None):
            
        if mask != None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        
        # Target token
        fus1 = self.cross_attention(x_g.unsqueeze(1), x_t)
            
        # Neighbor token
        fus2 = self.cross_attention(x_g.unsqueeze(1), x_n, mask=mask) 
                
        fusion = (fus1 + fus2).squeeze(1)            
        fusion = self.norm(fusion)
        
        return fusion



class CrossFusion(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.cross_attention = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x1, x2, mask=None):
        if mask != None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        
        fusion = self.cross_attention(x1, x2, mask=mask)
        fusion = self.norm(fusion)

        return fusion




class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, output_fmt='BNC'):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        if output_fmt != 'BCHW':
            self.flatten = True
        else:
            self.flatten = False


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    


class ImageEncoder_SWIN_TRANS(nn.Module):
    def __init__(
        self, model_name="swin_tiny_patch4_window7_224", pretrained=False, trainable=True,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, embed_layer=ConvStem, pretrained=pretrained
        )
        self.model.head = nn.Identity()
        self.model.avgpool = nn.Identity()
        msg=self.model.load_state_dict(torch.load(r'weights\ctranspath.pth')['model'], strict=True)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        features = self.model(x)
        return features[-1]
    



def load_model_weights(model_name:str, path: str):
    model = torchvision.models.__dict__[model_name](weights=None)

    state_dict = torch.load(path)

    if model_name == "resnet18":
        state_dict = state_dict['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.resnet.', '')] = state_dict.pop(key)

    if model_name == "densenet121":
        for key in list(state_dict.keys()):
            state_dict[key.replace('module.model.0', 'features').replace('module.fc_4', 'classifier')] = state_dict.pop(key)

    model_dict = model.state_dict()
    for key in list(model_dict.keys())[-2:]:
        del model_dict[key]

    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    if state_dict == {}:
        print('No weight could be loaded..')
    model_dict.update(state_dict)
    model.load_state_dict(model_dict, strict=False)
    
    if model_name == "resnet18":
        model.fc = nn.Identity()
    if model_name == "densenet121":
        model.classifier = nn.Identity()

    return model