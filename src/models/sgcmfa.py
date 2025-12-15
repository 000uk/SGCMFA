import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.graph_layers import Graph, TemporalAttention, ST_GCN_Block
from torchvision.models import mobilenet_v3_small

class RGB_Encoder(nn.Module): # ★ 2D CNN (MobileNet)
    def __init__(self):
        super(RGB_Encoder, self).__init__()
        self.backbone = mobilenet_v3_small(pretrained=True).features
        self.conv_project = nn.Conv2d(576, 128, kernel_size=1) 

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        x = self.backbone(x)     # (B*T, 576, 7, 7)
        x = self.conv_project(x) # (B*T, 128, 7, 7)
        _, C_new, H_new, W_new = x.shape
        x = x.view(B, T, C_new, H_new, W_new).permute(0, 2, 1, 3, 4) 
        return x # (B, 128, T, 7, 7)

class Skeleton_Encoder(nn.Module): # ★ ST-GCN Part
    def __init__(self, in_channels, out_channels, A):
        super(Skeleton_Encoder, self).__init__()
        self.gcn_networks = nn.Sequential(
            ST_GCN_Block(in_channels, 64, A),
            ST_GCN_Block(64, 64, A),
            ST_GCN_Block(64, 128, A) # 128채널로 맞춰줌
        )
    def forward(self, x):
        for gcn in self.gcn_networks: x = gcn(x)
        return x

class CMFA_Fusion(nn.Module): # ★ 핵심: 좌표로 RGB 필터링
    def __init__(self, channel=128):
        super(CMFA_Fusion, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, skel_feat, rgb_feat):
        B, C, T, V = skel_feat.shape
        _, _, _, H, W = rgb_feat.shape
        
        Q = skel_feat.permute(0, 2, 3, 1).contiguous().view(B, T, V, C)
        K = rgb_feat.view(B, C, T, H*W).permute(0, 2, 1, 3)
        
        attn = self.softmax(torch.matmul(Q, K)) # (B, T, V, HW) - Attention Map
        V_rgb = rgb_feat.view(B, C, T, H*W).permute(0, 2, 3, 1)
        
        out = torch.matmul(attn, V_rgb).permute(0, 3, 1, 2).contiguous() # (B, C, T, V)
        return self.out_conv(out) + skel_feat # Residual

class SGCMFA_Net(nn.Module):
    def __init__(self, num_classes=5):
        super(SGCMFA_Net, self).__init__()
        self.graph = Graph()
        A = self.graph.A
        
        # 1. Encoders
        self.rgb_stream = RGB_Encoder()
        self.skel_stream = Skeleton_Encoder(3, 128, A)
        
        # 2. Fusion
        self.fusion = CMFA_Fusion(channel=128)
        
        # 3. Backend (Deep Features)
        self.backend = nn.Sequential(
            ST_GCN_Block(128, 256, A, stride=2),
            ST_GCN_Block(256, 256, A)
        )
        
        # 4. Global Temporal Attention (★ 추가됨: Global Refinement)
        self.temporal_attn = TemporalAttention(256)
        
        # 5. Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x_rgb, x_skel):
        # x_rgb: (B, 3, T, H, W)
        # x_skel: (B, T, V, C) -> Preprocessing expected: (B, C, T, V) and Normalized
        
        N, T, V, C = x_skel.size()
        x_skel = x_skel.permute(0, 3, 1, 2).contiguous() 
        
        # Encoding
        feat_rgb = self.rgb_stream(x_rgb) 
        feat_skel = self.skel_stream(x_skel)
        
        # Fusion (Skeleton Guides RGB)
        x = self.fusion(feat_skel, feat_rgb)
        
        # Backend Processing
        for block in self.backend: x = block(x)
        
        # ★ Global Temporal Gating (여기가 추가된 핵심)
        # 중요하지 않은 프레임(준비/원위치 동작)을 마스킹
        x = self.temporal_attn(x)
        
        # Classification
        x = F.avg_pool2d(x, x.size()[2:]).view(N, -1)
        return self.fc(x)