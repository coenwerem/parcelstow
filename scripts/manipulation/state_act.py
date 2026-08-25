"""State-only ACT model (Zhao et al., RSS 2023), the released DETRVAE with
the image branch removed.

The released model (detr/models/detr_vae.py) conditions the transformer on
two extra tokens, the CVAE latent and the projected proprioception, in front
of the image tokens, and its state-only branch is a stub. The class below
reproduces the released forward pass for the token sequence [latent,
proprio] alone, with the same CVAE encoder over [CLS, qpos, action chunk],
the same sinusoidal table, the same reparametrization, the same learned
additional position embeddings, and the released Transformer, encoder, and
decoder classes (vendored under third_party/act). State and action
dimensions are arguments, the release hard-codes 14 for both.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from third_party.act.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.FloatTensor(table).unsqueeze(0)


def kl_divergence(mu, logvar):
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    return total_kld


def build_encoder(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, pre_norm=False):
    encoder_layer = TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, "relu", pre_norm)
    encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
    return TransformerEncoder(encoder_layer, enc_layers, encoder_norm)


class StateACT(nn.Module):
    """DETRVAE forward for the token sequence [latent, proprio]."""

    def __init__(self, state_dim, action_dim, num_queries, hidden_dim=512, dim_feedforward=3200,
                 enc_layers=4, dec_layers=7, nheads=8, dropout=0.1, latent_dim=32):
        super().__init__()
        self.num_queries = num_queries
        self.latent_dim = latent_dim
        self.transformer = Transformer(
            d_model=hidden_dim, dropout=dropout, nhead=nheads, dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, normalize_before=False,
            return_intermediate_dec=True,
        )
        self.encoder = build_encoder(hidden_dim, dropout, nheads, dim_feedforward, enc_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)
        self.register_buffer("pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, qpos, actions=None, is_pad=None):
        is_training = actions is not None
        bs = qpos.shape[0]
        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1).permute(1, 0, 2)
            cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
            is_pad_full = torch.cat([cls_joint_is_pad, is_pad], dim=1)
            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad_full)[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_input = self.latent_out_proj(reparametrize(mu, logvar))
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
        proprio_input = self.input_proj_robot_state(qpos)
        # the released image branch, minus the image tokens
        src = torch.stack([latent_input, proprio_input], dim=0)  # (2, bs, hidden)
        pos_embed = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        memory = self.transformer.encoder(src, pos=pos_embed)
        hs = self.transformer.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)[0]  # first decoder output stack entry as in the release
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, (mu, logvar)
