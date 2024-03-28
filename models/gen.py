import copy
from typing import Optional, List

import einops
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from functools import partial

from mamba_ssm import Mamba
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class GEN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        instance_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                         dropout, activation, normalize_before)
        instance_decoder_norm = nn.LayerNorm(d_model)
        self.instance_decoder = TransformerDecoder(instance_decoder_layer,
                                                   num_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)

        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before)
        interaction_decoder_norm = nn.LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer,
                                                      num_dec_layers,
                                                      interaction_decoder_norm,
                                                      return_intermediate=return_intermediate_dec)
        ##gated attention
        self.BiAttentionBlock = BiAttentionBlock(d_model, d_model, d_model, nhead)
        self.linear_layer = nn.Sequential(
                nn.Linear(2*d_model, d_model),
                nn.LayerNorm(d_model),
            )
        self.llm_softmax = nn.Softmax(dim=-1)
        ##mamba
        self.mambalayer = MambaLayer(dim=d_model)
        
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward_llm(self, text_embedding, union_feature):
        x = torch.matmul(union_feature, text_embedding.transpose(2, 3))  # torch.Size([LB, H*W, N])
        x = self.llm_softmax(x)  # torch.Size([L, B, K, N])
        x = torch.matmul(x, text_embedding) # torch.Size([LB, H*W, D])
        return x
    
    # torch.Size([4, 256, 24, 42]) #pos_embed:torch.Size([4, 256, 24, 42]) #temp_feat: torch.Size([4, 256, 3, 24, 42])
    def forward(self, src, mask, query_embed_h, query_embed_o, pos_guided_embed, pos_embed, clip_feature, temp_feat):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape # 4, 256, 24, 42
        src = src.flatten(2).permute(2, 0, 1)
        
        pos_temp = pos_embed
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) #torch.Size([1008, 4, 256])
        num_queries = query_embed_h.shape[0]
        
        query_embed_o = query_embed_o + pos_guided_embed
        query_embed_h = query_embed_h + pos_guided_embed
        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)
        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)

        mask = mask.flatten(1)
        ins_tgt = torch.zeros_like(ins_query_embed)
        
        mamba_pos = 'before'
        if mamba_pos == 'before':
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) #torch.Size([1008, 4, 256])
            ##addpos
            # pos_temp = pos_temp.unsqueeze(0).repeat(3, 1, 1, 1, 1).permute(1, 2, 0, 3, 4)
            # temp_feat = pos_temp + temp_feat
            updated_memory = self.mambalayer(temp_feat)
            updated_memory = updated_memory[:, :, 1:2, :, :]
            updated_memory = updated_memory.view(bs, c, h*w).permute(2, 0, 1)        
        elif mamba_pos == 'after':
            # mamba
            temp_feat = temp_feat.permute(2, 0, 1, 3, 4) #torch.Size([3, 4, 256, 24, 42]) 
            temp_feat = einops.rearrange(temp_feat, 't b c h w -> (t b) c h w')
            temp_feat = temp_feat.flatten(2).permute(2, 0, 1) #torch.Size([1008, 12, 256])
            temp_mask = mask.repeat(3, 1) #torch.Size([12, 1008])
            temp_pos_embed = pos_embed.repeat(1, 3, 1) #torch.Size([1008, 12, 256])
            temp_memory = self.encoder(temp_feat, src_key_padding_mask=temp_mask, pos=temp_pos_embed) #[hw, 12, 256] #torch.Size([1008, 12, 256])
            memory = temp_memory[:,bs:2*bs,:] #torch.Size([1008, 4, 256])
            temp_memory = temp_memory.permute(1, 2, 0).view(bs, 3, c, -1) #torch.Size([4, 3, 256, 24*42])
            temp_memory = temp_memory.view(bs, 3, c, h, w) #torch.Size([4, 3, 256, 24, 42]) b, c, t, h ,w
            updated_temp_feat = self.mambalayer(temp_memory.permute(0, 2, 1, 3, 4)) #torch.Size([4, 256, 3, 24, 42])
            updated_temp_feat = updated_temp_feat[:, :, 1:2, :, :].squeeze(2) #torch.Size([4, 256, 24, 42])
            updated_memory = updated_temp_feat.view(bs, c, h*w).permute(2, 0, 1) #torch.Size([1008, 4, 256])
        else:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) #torch.Size([1008, 4, 256])
            updated_memory = memory
        
        ## ToDo: Semantic_spatial_feature
        visual_feature = memory.permute(1, 0, 2) #torch.Size([4, 1008, 256])
        clip_feature = self.linear_layer(clip_feature.unsqueeze(0).repeat(bs, 1, 1)) # torch.Size([4, 100, 256])
        s_sv, s_sl = self.BiAttentionBlock(visual_feature, clip_feature)
        s_sv = s_sv.permute(1, 0, 2) #torch.Size([1008, 4, 256])
        
        ins_hs = self.instance_decoder(ins_tgt, memory, memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)
        ins_hs = ins_hs.transpose(1, 2)
        h_hs = ins_hs[:, :, :num_queries, :]
        o_hs = ins_hs[:, :, num_queries:, :] #torch.Size([3, 4, 100, 256])

        # add
        ins_guided_embed = (h_hs + o_hs) / 2.0
        
        clip_way = 'add'
        if clip_way == 'add':
            ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3) #torch.Size([3, 100, 4, 256])
            updated_feature = s_sl.permute(1, 0, 2) #torch.Size([100, 4, 256])
            updated_feature = torch.unsqueeze(updated_feature, dim=0) 
            updated_feature = updated_feature.expand(3, -1, -1, -1) #torch.Size([3, 100, 4, 256])
            ins_guided_embed = updated_feature + ins_guided_embed #torch.Size([3, 100, 4, 256])
        elif clip_way == 'mul':
            updated_feature = s_sl 
            updated_feature = torch.unsqueeze(updated_feature, dim=0) 
            updated_feature = updated_feature.expand(3, -1, -1, -1) #torch.Size([3, 4, 100, 256])
            ins_guided_embed = self.forward_llm(updated_feature, ins_guided_embed) #torch.Size([3, 4, 100, 256])
            ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3) #torch.Size([3, 100, 4, 256])
        
        inter_tgt = torch.zeros_like(ins_guided_embed[0]) #torch.Size([100, 4, 256])
        inter_hs = self.interaction_decoder(inter_tgt, updated_memory, memory_key_padding_mask=mask,
                                            pos=pos_embed, query_pos=ins_guided_embed)
        inter_hs = inter_hs.transpose(1, 2) #torch.Size([3, 4, 100, 256])
        
        return h_hs, o_hs, inter_hs, memory.permute(1, 2, 0).view(bs, c, h, w)
    
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, nf, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, nf, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, nf, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, nf, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, nf, H, W = x.shape

        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2) #torch.Size([4, 3024, 256])

        # wrong:stack expects each tensor to be equal size, but got [4, 1024, 605] at entry 0 and [4, 1024, 604] at entry 4
        norm1 = self.norm1(x_flat) #torch.Size([4, 3024, 256])
        mamba1 = self.mamba(norm1)
        droppath = self.drop_path(mamba1)
        x_mamba = x_flat + droppath
        x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos) #torch.Size([1008, 12, 256])
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_gen(args):
    return GEN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

## ToDo: GatedCrossAttention
class RLIPv2_BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1):
        super(RLIPv2_BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim #256
        self.num_heads = num_heads #8
        self.head_dim = embed_dim // num_heads #32
        self.v_dim = v_dim 
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5) 
        self.dropout = dropout #0.1

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim) #Linear(in_features=256, out_features=256, bias=True)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim) #Linear(in_features=256, out_features=256, bias=True)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim) #Linear(in_features=self.v_dim, out_features=256, bias=True)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim) #Linear(in_features=self.l_dim, out_features=256, bias=True)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim) #Linear(in_features=256, out_features=self.v_dim, bias=True)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim) #Linear(in_features=l_dim, out_features=self.l_dim, bias=True)
        
        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, v, l, v_pos=None, attention_mask_l=None, attention_mask_v=None):
        # TODO:
        # Add with_pos_embed(v, v_pos) for attention calculation
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(self.with_pos_embed(v, v_pos)) * self.scale  # bsz, tgt_len, embed_dim    torch.Size([4, HW, 256])
        key_states = self._shape(self.l_proj(l), -1, bsz)  # bsz, self.num_heads, -1, self.head_dim  torch.Size([4, 8, 100, 32])
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)  # bsz, self.num_heads, -1, self.head_dim #torch.Size([4, 8, HW, 32])
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)  # bsz, self.num_heads, -1, self.head_dim #torch.Size([4, 8, 100, 32])

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # bsz * self.num_heads, -1, self.head_dim #torch.Size([32, HW, 32])
        key_states = key_states.view(*proj_shape) # bsz * self.num_heads, -1, self.head_dim torch.Size([32, 100, 32]) 
        value_v_states = value_v_states.view(*proj_shape) # bsz * self.num_heads, -1, self.head_dim torch.Size([32, HW, 32])
        value_l_states = value_l_states.view(*proj_shape) # bsz * self.num_heads, -1, self.head_dim torch.Size([32, 100, 32]) 

        src_len = key_states.size(1) # 100
        #batch matrix multiplication
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) 
        # bsz * self.num_heads, tgt_len, src_len torch.Size([32, HW, 100]) 

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)
        attn_weights_T = attn_weights.transpose(1, 2) # torch.Size([32, 100, HW])
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]) # torch.Size([32, 100, HW])
        
        # Following lines (if) are added to mask out visual features for the calculation of attn_weights_l
        if attention_mask_v is not None:
            # shape of attn_weights_l: [bsz, src_len, tgt_len]  src_len = lang_len; tgt_len = vis_len
            assert (attention_mask_v.dim() == 2)
            attention_mask = attention_mask_v.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, src_len, tgt_len) #torch.Size([4, 1, 100, 1025])
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, src_len, tgt_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, src_len, tgt_len)}"
                )
            attn_weights_l = attn_weights_l.view(bsz, self.num_heads, src_len, tgt_len) + attention_mask
            attn_weights_l = attn_weights_l.view(bsz * self.num_heads, src_len, tgt_len)
        #######################

        attn_weights_l = attn_weights_l.softmax(dim=-1) #torch.Size([32, 100, HW])

        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1) #torch.Size([32, HW, 100])

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training) #torch.Size([32, HW, 100])
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training) #torch.Size([32, 100, HW])

        attn_output_v = torch.bmm(attn_probs_v, value_l_states) # torch.Size([32, HW, 32])
        attn_output_l = torch.bmm(attn_probs_l, value_v_states) # torch.Size([32, 100, 32])


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim) # torch.Size([4, 8, HW, 32])
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim) # torch.Size([4, HW, 256])

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim) # torch.Size([4, 8, 100, 32])
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim) # torch.Size([4, 100, 256])

        attn_output_v = self.out_v_proj(attn_output_v) # torch.Size([4, 1008, 256])
        attn_output_l = self.out_l_proj(attn_output_l) # torch.Size([4, 56, 256])

        return attn_output_v, attn_output_l

# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim) #LayerNorm((v_dim,), eps=1e-05, elementwise_affine=True)
        self.layer_norm_l = nn.LayerNorm(l_dim) #LayerNorm((l_dim,), eps=1e-05, elementwise_affine=True)
        self.attn = RLIPv2_BiMultiHeadAttention(v_dim=v_dim,
                                                l_dim=l_dim,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True) #torch.Size([v_dim])
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True) #torch.Size([l_dim])

    def forward(self, v, l, v_pos=None, attention_mask_l=None, attention_mask_v=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, v_pos, attention_mask_v=attention_mask_v) #torch.Size([4, HW, 256]) #torch.Size([4, 100, 256])
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v) #torch.Size([4, HW, 256])
        l = l + self.drop_path(self.gamma_l * delta_l) #torch.Size([4, 100, 256])
        return v, l