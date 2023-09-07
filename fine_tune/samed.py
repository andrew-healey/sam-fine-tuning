from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic

from typing import Any, Dict, List, Tuple, Type


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

from segment_anything.modeling.mask_decoder import MaskDecoder

class _LoRA_qkv_proj(nn.Module):
    def __init__(self, proj: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.proj = proj
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.proj(x) + self.w_b(self.w_a(x))
        return x

class LoRA_Mask_Decoder(nn.Module):
    """Applies low-rank adaptation to a Sam model's mask decoder.

    Args:
        mask_decoder: a mask decoder model
        r: rank of LoRA

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(
            self,
            mask_decoder: MaskDecoder, r: int):
        super(LoRA_Mask_Decoder, self).__init__()

        assert r > 0
        self.r = r

        # Surgery for the mask decoder
        for param in mask_decoder.transformer.parameters():
            param.requires_grad = False

        self.self_attn = nn.ModuleList()
        self.cross_attn_ti = nn.ModuleList()
        self.cross_attn_it = nn.ModuleList()

        decoder_transformer = mask_decoder.transformer
        for blk in decoder_transformer.layers:
            self.self_attn.append(self.lorify_qkv_proj(blk.self_attn))
            self.cross_attn_ti.append(self.lorify_qkv_proj(blk.cross_attn_token_to_image))
            self.cross_attn_it.append(self.lorify_qkv_proj(blk.cross_attn_image_to_token))
        
        self.final_attn = self.lorify_qkv_proj(decoder_transformer.final_attn_token_to_image)

        # self.reset_parameters()
        self.mask_decoder = mask_decoder
    
    def lorify_qkv_proj(self, attn: nn.Module):
        cust_state_dict = nn.ParameterDict()

        q_proj = attn.q_proj
        v_proj = attn.v_proj

        in_dim, out_dim = attn.embedding_dim, attn.internal_dim

        # A and B for query
        w_a_linear_q = nn.Linear(in_dim, self.r, bias=False)
        w_b_linear_q = nn.Linear(self.r, out_dim, bias=False)

        # A and B for value
        w_a_linear_v = nn.Linear(in_dim, self.r, bias=False)
        w_b_linear_v = nn.Linear(self.r, out_dim, bias=False)

        cust_state_dict["a_q"] = w_a_linear_q.weight
        cust_state_dict["b_q"] = w_b_linear_q.weight
        cust_state_dict["a_v"] = w_a_linear_v.weight
        cust_state_dict["b_v"] = w_b_linear_v.weight

        # Error: torch.FloatTensor is not a Module subclass


        attn.q_proj = _LoRA_qkv_proj(q_proj, w_a_linear_q, w_b_linear_q)
        attn.v_proj = _LoRA_qkv_proj(v_proj, w_a_linear_v, w_b_linear_v)

        return cust_state_dict
    
    def save_lora_parameters(self,filename:str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        raw_state_dict = self.state_dict()
        # remove mask_decoder state_dict
        state_dict = {
            k: v for k, v in raw_state_dict.items() if 'mask_decoder' not in k
        }

        torch.save(state_dict, filename)
    
    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)
    
    def get_parameters(self) -> List[Parameter]:
        ret = []
        ret.extend(self.self_attn.parameters())
        ret.extend(self.cross_attn_ti.parameters())
        ret.extend(self.cross_attn_it.parameters())
        ret.extend(self.final_attn.parameters())

        return ret

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)

from segment_anything.modeling.tiny_vit_sam import TinyViT


class _LoRA_Tiny_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, h, bias=qkv_bias)
    B, N, _ = x.shape
    qkv = self.qkv(x)
    q, k, v = qkv.view(B, N, self.num_heads, -
                        1).split([self.key_dim, self.key_dim, self.d], dim=3)
    """

    def __init__(
            self,
            qkv: nn.Module,
            key_dim: int,
            num_heads: int,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            h: int,
    ):
        super().__init__()
        self.qkv = qkv

        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q

        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v

        self.key_dim = key_dim
        self.num_heads = num_heads
        self.h = h

    def forward(self, x):
        B,N,_ = x.shape
        qkv = self.qkv(x)  # B,N,N,3*org_C
        qkv = qkv.view(B,N,self.num_heads,-1)

        new_q = self.linear_b_q(self.linear_a_q(x))
        new_q = new_q.view(B,N,self.num_heads,-1)
        new_v = self.linear_b_v(self.linear_a_v(x))
        new_v = new_v.view(B,N,self.num_heads,-1)

        qkv[:, :, :, :self.key_dim] += new_q
        qkv[:, :, :, 2*self.key_dim:] += new_v

        return qkv.view(B,N,-1)


class LoRA_Tiny_Image_Encoder(nn.Module):
    def __init__(self,image_encoder:TinyViT,r:int):
        super(LoRA_Tiny_Image_Encoder,self).__init__()

        assert isinstance(image_encoder,TinyViT), "Only TinyViT is supported now for LoRA."

        assert r > 0
        self.r = r

        # Surgery for the image encoder
        for param in image_encoder.parameters():
            param.requires_grad = False
        
        self.self_attn = nn.ModuleList()

        for layer in image_encoder.layers[1:]:
            sub_list = nn.ModuleList()
            for blk in layer.blocks:
                sub_list.append(self.lorify_attn(blk.attn))
            self.self_attn.append(sub_list)
        
        self.image_encoder =  image_encoder
    
    def lorify_attn(self, attn: nn.Module):
        cust_state_dict = nn.ParameterDict()

        key_dim = attn.key_dim
        nh_kd = attn.nh_kd
        dh = attn.dh
        num_heads = attn.num_heads
        dim = attn.qkv.in_features
        h = attn.h

        linear_a_q = nn.Linear(dim, self.r, bias=False)
        linear_b_q = nn.Linear(self.r, nh_kd, bias=False)

        linear_a_v = nn.Linear(dim, self.r, bias=False)
        linear_b_v = nn.Linear(self.r, dh, bias=False)

        cust_state_dict["a_q"] = linear_a_q.weight
        cust_state_dict["b_q"] = linear_b_q.weight
        cust_state_dict["a_v"] = linear_a_v.weight
        cust_state_dict["b_v"] = linear_b_v.weight

        attn.qkv = _LoRA_Tiny_qkv(
            attn.qkv, key_dim, num_heads,
            linear_a_q, linear_b_q, linear_a_v, linear_b_v,
            h,
        )

        return cust_state_dict

    def save_lora_parameters(self,filename:str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        raw_state_dict = self.state_dict()
        # remove mask_decoder state_dict
        state_dict = {
            k: v for k, v in raw_state_dict.items() if 'image_encoder' not in k
        }

        torch.save(state_dict, filename)
    
    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def get_trainable_parameters(self) -> List[Parameter]:
        return self.self_attn.parameters()
    
    def get_trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('image_encoder')}

    def forward(self, *args, **kwargs):
        return self.image_encoder(*args, **kwargs)

