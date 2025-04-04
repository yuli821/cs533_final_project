import torch
import torch.nn as nn
from torch.nn import Dropout
import copy

class myPatchEmbed(nn.Module) :
    '''
    Input embedding, divide inputs into tokens
    '''
    def __init__(self, img_size=32, patch_size=4, in_chans=3, d_model=32):
        super().__init__()
        assert img_size % patch_size == 0    #ensure image is divisible by patches
        self.conv = nn.Conv2d(in_channels=in_chans,
                              out_channels=d_model,
                              kernel_size=patch_size,
                              stride=patch_size,)
        # num_patches = (img_size//patch_size)**2

    def forward(self, x) :
        out = self.conv(x)    #(B, D, H/P, W/P)
        out = torch.flatten(out, start_dim=2)   #(B, D, HW/P^2)
        out = out.transpose(1,2)    #(B, HW/P^2, D)
        return out

class myMlp(nn.Module) :
    '''
    Designed for a single hidden layer
    '''
    def __init__(self, in_features=16, hidden_features=64, out_features=16, act_layer=nn.functional.gelu):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = act_layer
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.dropout = Dropout(0.1)

    def forward(self, x) :
        out = self.linear1(x)
        out = self.linear2(self.dropout(self.act(out)))
        out = self.dropout(out)
        return out

class myAttention(nn.Module) :
    '''
    Multi-Headed Attention
    '''
    def __init__(self, dim=32, nhead=4, qkv_bias=True):
        super().__init__()
        assert dim % nhead == 0
        self.headdim = dim//nhead
        self.scale = (self.headdim)**(-0.5)
        self.nhead = nhead
        self.qkv = nn.Linear(in_features=dim, out_features=int(3*dim),    #Wq, Wk, Wv matrices (D, D)
                             bias=qkv_bias)
        self.proj = nn.Linear(in_features=dim, out_features=dim)          #combining multiple heads (D, D)

    def forward(self, x) :
        B, N, D = x.shape
        qkv = self.qkv(x)                                                          #(B, N, 3D)        dpdim: D
        qkv = qkv.reshape(B, N, 3, self.nhead,                        #reshape: (B, N, 3, H, Dh)
                          self.headdim).permute(2, 0, 3, 1, 4)           #permute: (3, B, H, N, Dh)
        q, k, v = qkv.unbind(0)    #separate Q, K, V                                (B, H, N, Dh)
        #scaled dot product attention
        q = q * self.scale    #scalar product
        attn = q @ k.transpose(-2, -1)    #QK^T                                     (B, H, N, N)      dpdim: Dh
        attn = attn.softmax(dim=-1)
        attn = attn @ v                                                            #(B, H, N, Dh)     dpdim: N
        out = attn.transpose(1, 2).reshape(B, N, D)   #concatenate the heads        (B, N, D)
        out = self.proj(out)                                                       #(B, N, D)         dpdim: D

        return out
    
class vit(nn.Module) :
    '''
    A small Vision Transformer (ViT)
    '''
    def __init__(self, ipch=1, Nclasses=10, image_size=32, d_model=32, num_layers=12, nhead=4, patch_size=4, mlp_ratio=2.0) :
        '''
        Input:
        ipch - no. of input channels
        Nclasses - no. of classes in input
        image_size - pixel dimension
        d_model - token dimension from embedding output
        layers - no. of layers
        config - output channels for each layer
        '''
        super().__init__()
        #library v/s self-implemented version
        attention = myAttention
        mlp = myMlp
        patchembed  = myPatchEmbed
        ffn_hidden = int(mlp_ratio*d_model)    #hidden dimension of MLP layer
        #--------------------------- First Layer ------------------------------#
        num_patches = int(image_size**2/patch_size**2)
        self.embedding = patchembed(img_size=image_size,    #convert image to tokens
                                    patch_size=patch_size,
                                    in_chans=ipch,
                                    d_model=d_model)                                #patch embedding
        self.classembed = nn.Parameter(torch.zeros(1, 1, d_model))                    #label embedding
        self.posencode = nn.Parameter(torch.zeros(1, num_patches+1, d_model))         #positional encoding
        self.dropout = Dropout(0.1)
        self.blocks = nn.ModuleList()
        #--------------------------- Transformer Block ------------------------------#
        # self.norm1 = nn.LayerNorm(d_model)                                                     #layernorm
        # self.attn1 = attention(dim=d_model, nhead=nhead, qkv_bias=True)                    #attention
        # self.norm2 = nn.LayerNorm(d_model)                                                     #layernorm
        # self.ffn1 = mlp(in_features=d_model, hidden_features=ffn_hidden, out_features=d_model, act_layer=nn.ReLU)    #mlp
        for _ in range(num_layers):
            block = nn.Sequential(
                    nn.LayerNorm(d_model, eps=1e-6),
                    attention(dim=d_model, nhead=nhead, qkv_bias=True),
                    nn.LayerNorm(d_model, eps=1e-6),
                    mlp(in_features=d_model, hidden_features=ffn_hidden, out_features=d_model, act_layer=nn.functional.gelu))
            self.blocks.append(copy.deepcopy(block))

        #--------------------------- Last Layer ------------------------------#
        self.fc = nn.Linear(in_features=d_model, out_features=Nclasses)

    def forward(self, x) :
        out = self.embedding(x)                                 #(B, Np, D)
        classtok = self.classembed.expand(x.size(0), -1, -1)    #change from (1, D) to (B, 1, D)
        out = torch.cat((classtok, out), dim=1)                 #(B, Np+1, D)
        out = out + self.posencode
        out = self.dropout(out)

        # for _ in range(num_layers):
        #     out = out + self.attn1(self.norm1(out))    #residual connection
        #     out = out + self.ffn1(self.norm2(out))    #residual connection
        for block in self.blocks:
            norm1, attn, norm2, ffn = block
            out = out + attn(norm1(out))
            out = out + ffn(norm2(out))

        out = out[:,0]    #connect classifier head only to the class token embedding
        # print(out.shape)
        out = self.fc(out)
        # print(out.shape)

        return out
    
