import torch
import torch.nn as nn
import torch.nn.functional as F

class CMFA_Fusion(nn.Module): # 좌표로 RGB 필터링
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

class DualFusionLayer(nn.Module):
    """
    Query(Learnable) -> Cross(Ref:Skeleton) -> Self -> Cross(Mem:RGB) -> FFN
    """
    def __init__(self, d_model=128, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # 1. Cross Attention 1: Query가 Skeleton(Ref)을 봄
        self.cross_attn_skel = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Self Attention: Query끼리 정보 교환
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. Cross Attention 2: Query가 RGB(Memory)를 봄
        self.cross_attn_rgb = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 4. Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, query, key_skel, val_skel, key_rgb, val_rgb):
        # 1. Skeleton 참조 (Pre-LN 적용)
        q = self.norm1(query)
        # query가 skeleton을 보며 내 관절 위치가 여기구나!!!!!! 파악함
        # feat_skel, _ = self.cross_attn_skel(q, key_skel, val_skel)
        feat_skel, attn_skel = self.cross_attn_skel(q, key_skel, val_skel) # attn_skel 추출
        query = query + feat_skel # Residual
        
        # 2. Self Attention
        q = self.norm2(query)
        # feat_self, _ = self.self_attn(q, q, q)
        feat_self, attn_self = self.self_attn(q, q, q) # attn_self 추출
        query = query + feat_self
        
        # 3. RGB 참조
        q = self.norm3(query)
        # 위치를 파악한 query가 RGB 텍스처를 보며 헐 손 모양이 이렇네~~ 파악
        # feat_rgb, _ = self.cross_attn_rgb(q, key_rgb, val_rgb)
        feat_rgb, attn_rgb = self.cross_attn_rgb(q, key_rgb, val_rgb) # attn_rgb 추출
        query = query + feat_rgb
        
        # 4. FFN
        q = self.norm4(query)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(q))))
        query = query + ffn_out
        
        return query, (attn_skel, attn_self, attn_rgb)

class DualFusionTransformer(nn.Module):
    def __init__(self, d_model=128, num_queries=21, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries # 관절 수
        
        # Learnable Queries: 각각의 쿼리가 다 관절 담당임
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            DualFusionLayer(d_model, nhead) for _ in range(num_layers)
        ])
        
        # Positional Encoding for RGB (7x7 spatial)
        self.pos_embed_rgb = nn.Parameter(torch.randn(1, 49, d_model)) 

    def forward(self, skel_feat, rgb_feat):
        # skel_feat: (B, C, T, V)
        # rgb_feat: (B, C, T, H, W) -> (B, 128, T, 7, 7)
        
        B, C, T, V = skel_feat.shape
        _, _, _, H, W = rgb_feat.shape
        
        # 처리를 위해 Batch와 Time을 합침 (Spatial Attention에 집중!!!)
        # (B*T, V, C) 형태로 변환
        feat_skel = skel_feat.permute(0, 2, 3, 1).contiguous().view(B*T, V, C)
        
        # (B*T, H*W, C) 형태로 변환
        feat_rgb = rgb_feat.view(B, C, T, H*W).permute(0, 2, 3, 1).contiguous().view(B*T, H*W, C)
        
        # Learnable Query 준비 (Batch*Time 만큼 복제)
        query = self.query_embed.weight.unsqueeze(0).repeat(B*T, 1, 1) # (B*T, 21, 128)
        
        # RGB에 위치 정보(Positional Embedding) 더하기
        feat_rgb = feat_rgb + self.pos_embed_rgb
        
        # Transformer 통과
        for layer in self.layers:
            # Query가 Skeleton을 Key/Value로 참조하고, 그 다음 RGB를 Key/Value로 참조
            query, attn_map = layer(query, feat_skel, feat_skel, feat_rgb, feat_rgb)
            
        # 원래 형태로 복원: (B*T, V, C) -> (B, T, V, C) -> (B, C, T, V)
        out = query.view(B, T, V, C).permute(0, 3, 1, 2).contiguous()
        
        # Residual Connection (원본 스켈레톤 특징 더해주기)
        return out + skel_feat, attn_map