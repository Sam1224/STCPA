import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from base import BaseModel
from .loss import mean_mse_loss


# ========================================
# DriveAttention
# A mix of three types of attention
# ========================================
class DriveAttention(nn.Module):
    def __init__(self, num_nodes, in_feature, n_temporal=3, device=None):
        super(DriveAttention, self).__init__()
        self.num_nodes = num_nodes
        self.in_feature = in_feature
        self.n_temporal = n_temporal
        self.device = device

        self.q_0_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)
        self.k_0_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)
        self.v_0_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)

        self.q_1_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)
        self.k_his_fc = nn.Linear(num_nodes, num_nodes * in_feature)
        self.v_his_fc = nn.Linear(num_nodes, num_nodes * in_feature)

        self.fuse_1_1_temporal_fc = nn.Linear(num_nodes * in_feature * n_temporal, num_nodes * in_feature)
        self.fuse_1_2_temporal_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)
        self.fuse_2_1_temporal_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)
        self.fuse_2_2_temporal_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)
        self.fuse_1_1_spatio_temporal_fc = nn.Linear(num_nodes * in_feature * 2, num_nodes * in_feature)
        self.fuse_1_2_spatio_temporal_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)
        self.fuse_2_1_spatio_temporal_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)
        self.fuse_2_2_spatio_temporal_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)

        self.q_2_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)
        self.k_2_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)
        self.v_2_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature)

        self.softmax = nn.Softmax(dim=-1)

        self.m_1_1_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature // 4)
        self.m_1_2_fc = nn.Linear(num_nodes * in_feature // 4, num_nodes * in_feature)
        self.m_his_1_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature // 4)
        self.m_his_2_fc = nn.Linear(num_nodes * in_feature // 4, num_nodes * in_feature)
        self.m_3_1_fc = nn.Linear(num_nodes * in_feature, num_nodes * in_feature // 4)
        self.m_3_2_fc = nn.Linear(num_nodes * in_feature // 4, num_nodes * in_feature)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, m_0, m_hiss):
        # m_0 => bs, num_edges * num_features
        # w_0 => bs, num_edges
        # m_hiss => bs, n_temporal, num_edges
        bs = m_0.size(0)
        num_edges = self.num_nodes

        # ========================================
        # Attention 1
        # Reconstruction
        # ========================================
        skip_0 = m_0.view(bs, -1)
        q_0 = self.q_0_fc(m_0).view(bs, num_edges, -1)  # bs, num_edges, num_features
        k_0 = self.k_0_fc(m_0).view(bs, num_edges, -1).permute(0, 2, 1)  # bs, num_features, num_edges
        v_0 = self.v_0_fc(m_0).view(bs, num_edges, -1)  # bs, num_edges, num_features

        energy_0 = torch.bmm(q_0, k_0)
        energy_0 = energy_0 / (self.in_feature ** 0.5)
        mask = torch.BoolTensor(np.expand_dims(np.eye(energy_0.size(1), dtype=int), axis=0)).to(
            self.device)  # do not need the element itself to reconstruct
        energy_0 = energy_0.masked_fill(mask, -np.inf)
        attention_0 = self.softmax(energy_0)  # bs, num_edges, num_edges

        m_1 = torch.bmm(attention_0, v_0)  # bs, num_edges, num_features
        m_1 = m_1.view(bs, -1)  # bs, num_edges * num_features

        m_1 = m_1 + skip_0  # bs, num_edges * num_features
        skip_1 = m_1  # bs, num_edges * num_features

        m_1 = F.relu(self.m_1_1_fc(m_1))  # bs, num_edges * num_features
        m_1 = self.m_1_2_fc(m_1)  # bs, num_edges * num_features
        m_1 = m_1 + skip_1  # bs, num_edges * num_features
        skip_1 = m_1  # bs, num_edges * num_features

        # ========================================
        # Attention 2
        # ========================================
        if self.n_temporal > 0:
            q_1 = self.q_1_fc(m_1).view(bs, num_edges, -1)  # bs, num_edges, num_features
            m_2s = []
            for idx in range(self.n_temporal):
                m_his = m_hiss[:, idx, ...]
                skip_his_1 = m_his
                k_1 = self.k_his_fc(m_his).view(bs, num_edges, -1).permute(0, 2, 1)  # bs, num_features, num_edges
                v_1 = self.v_his_fc(m_his).view(bs, num_edges, -1)  # bs, num_edges, num_features

                energy_1 = torch.bmm(q_1, k_1)
                energy_1 = energy_1 / (self.in_feature ** 0.5)

                # ========================================
                # By default, do not use mask and use TDA, if you want to use mask, then uncomment this code block
                # ========================================
                # # DA only
                # mask = torch.BoolTensor(np.expand_dims(np.eye(energy_1.size(1), dtype=int), axis=0)).to(self.device)
                # energy_1 = energy_1.masked_fill(mask, -np.inf)
                # ========================================

                attention_1 = self.softmax(energy_1)  # bs, num_edges, num_edges

                m_2 = torch.bmm(attention_1, v_1)  # bs, num_edges, num_features
                m_2 = m_2.view(bs, -1)  # bs, num_edges * num_features

                m_2 = m_2 + skip_his_1  # bs, num_edges * num_features
                skip_his_2 = m_2  # bs, num_edges * num_features

                m_2 = F.relu(self.m_his_1_fc(m_2))  # bs, num_edges * num_features
                m_2 = self.m_his_2_fc(m_2)  # bs, num_edges * num_features
                m_2 = m_2 + skip_his_2  # bs, num_edges * num_features

                m_2s.append(m_2)

            m_2 = torch.cat(m_2s, dim=-1)  # bs, num_edges * num_features * n_temporal

            m_hiss = F.relu(m_2.view(bs, -1, self.n_temporal).permute(0, 2, 1))  # bs, n_temporal, num_edges

            m_2 = F.relu(self.fuse_1_1_temporal_fc(m_2))  # bs, num_edges * num_features
            m_2 = self.fuse_1_2_temporal_fc(m_2)  # bs, num_edges * num_features
            skip_2 = m_2  # bs, num_edges * num_features

            m_2 = F.relu(self.fuse_2_1_temporal_fc(m_2))  # bs, num_edges * num_features
            m_2 = self.fuse_2_2_temporal_fc(m_2)  # bs, num_edges * num_features
            m_2 = m_2 + skip_2  # bs, num_edges * num_features
            skip_2 = m_2  # bs, num_edges * num_features

            m_fuse = torch.cat([skip_1, skip_2], dim=-1)  # bs, num_edges * num_features * 2
            m_fuse = F.relu(self.fuse_1_1_spatio_temporal_fc(m_fuse))  # bs, num_edges * num_features
            m_fuse = self.fuse_1_2_spatio_temporal_fc(m_fuse)  # bs, num_edges * num_features
            skip_2 = m_fuse  # bs, num_edges * num_features

            m_fuse = F.relu(self.fuse_2_1_spatio_temporal_fc(m_fuse))  # bs, num_edges * num_features
            m_fuse = self.fuse_2_2_spatio_temporal_fc(m_fuse)  # bs, num_edges * num_features
            m_fuse = m_fuse + skip_2  # bs, num_edges * num_features
        else:
            m_fuse = m_1  # bs, num_edges * num_features

        skip_2 = m_fuse  # bs, num_edges * num_features

        # ========================================
        # Attention 3
        # ========================================
        q_2 = self.q_2_fc(m_fuse).view(bs, num_edges, -1)  # bs, num_edges, num_features
        k_2 = self.k_2_fc(m_fuse).view(bs, num_edges, -1).permute(0, 2, 1)  # bs, num_features, num_edges
        v_2 = self.v_2_fc(m_fuse).view(bs, num_edges, -1)  # bs, num_edges, num_features

        energy_2 = torch.bmm(q_2, k_2)
        energy_2 = energy_2 / (self.in_feature ** 0.5)
        attention_2 = self.softmax(energy_2)  # bs_num_edges, num_edges

        m_3 = torch.bmm(attention_2, v_2)  # bs, num_edges, num_features
        m_3 = m_3.view(bs, -1)  # bs, num_edges * num_features
        m_3 = m_3 + skip_2
        skip_3 = m_3

        m_3 = F.relu(self.m_3_1_fc(m_3))
        m_3 = self.m_3_2_fc(m_3)
        m_3 = m_3 + skip_3
        m_3 = F.relu(m_3)
        return m_3, m_hiss


# ========================================
# STGAIN_FC
# ========================================
class STGAIN_FC(nn.Module):
    def __init__(self, num_nodes, n_blocks=5, n_temporal=3, device=None):
        super(STGAIN_FC, self).__init__()
        self.num_nodes = num_nodes  # 146 for chengdu
        self.n_blocks = n_blocks
        self.n_temporal = n_temporal
        self.device = device

        self.fc_1 = nn.Linear(2 * num_nodes, num_nodes)
        self.fc_2 = nn.Linear(num_nodes, num_nodes)
        self.fc_3 = nn.Linear(num_nodes, num_nodes)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, m, w):
        inputs = torch.cat([m, w], dim=1)
        m = F.relu(self.fc_1(inputs))
        m = F.relu(self.fc_2(m))
        m = self.fc_3(m)
        m_fc = self.sigmoid(m)
        return m_fc


# ========================================
# STGAIN_Att
# ========================================
class STGAIN_Att(nn.Module):
    def __init__(self, num_nodes, n_blocks=5, n_temporal=3, device=None):
        super(STGAIN_Att, self).__init__()
        self.num_nodes = num_nodes  # 146 for chengdu
        self.n_blocks = n_blocks
        self.n_temporal = n_temporal
        self.device = device

        self.in_feature = 1

        atts = []
        for idx in range(n_blocks):
            atts.append(
                DriveAttention(num_nodes, self.in_feature, n_temporal, device=device).to(device)
            )
        self.atts = nn.ModuleList(atts)
        self.att_fc_4 = nn.Linear(num_nodes, num_nodes)
        self.att_fc_5 = nn.Linear(num_nodes, num_nodes)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, m_att, m_his):
        bs = m_att.size(0)

        for idx, att in enumerate(self.atts):
            m_att, m_his = att(m_att, m_his)

        m_att = m_att.view(bs, -1)
        m_att = F.relu(self.att_fc_4(m_att))
        m_att = self.att_fc_5(m_att)
        m_att = self.sigmoid(m_att)
        return m_att
