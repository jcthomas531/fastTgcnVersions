# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:01:11 2026

@author: jthomas48
"""

def forward(self, x, index_face):
    # ------------------------------------------------------------
    # Adjacency (Graph) Construction  →  "Graph / Adjacency Matrix"
    # ------------------------------------------------------------
    adj = Adj_matrix_gen(index_face)         # build vertex adjacency from faces  [B, N, N]  
    adj = adj @ adj                          # enlarge receptive field (2-hop neighbors)  

    # ------------------------------------------------------------
    # Input Split  →  "Coordination (N×12)"  +  "Normal (N×12)"
    # (N = vertices; channel-first layout: [B, C, N])
    # ------------------------------------------------------------
    coor = x[:, :12, :]                      # raw coordination features (incoming N×12)  
    nor  = x[:, 12:, :]                      # raw normal features (incoming N×12)        

    # ------------------------------------------------------------
    # First per-stream encoders  →  "Coordination" & "Normal" boxes
    # These correspond to the orange/blue boxes BEFORE the GCN block.
    # After these 1×1 conv stacks, both streams are N×64.
    # ------------------------------------------------------------
    coor = self.conv1_c(coor)                # 12 → 64 channels  (Coordination encoder)   
    nor  = self.conv1_n(nor)                 # 12 → 64 channels  (Normal encoder)         
    # At this point: F_c = coor (N×64), F_n = nor (N×64) — the arrow labeled F_n
    # in the figure is this 'nor' tensor moving into the GCN.                     

    # ------------------------------------------------------------
    # Stage 1  →  "Graph Convolution Block"
    # (Note: The implementation applies GCNs to BOTH streams.)
    # Each GCN has a head: in→out (Conv1d), then several graph() blocks, then a tail + residual.
    # ------------------------------------------------------------
    coor1 = self.gcn_coor_1_1(coor, adj)     # 64 → 128 on coord stream           
    coor1 = self.gcn_coor_1_2(coor1, adj)    # 128 → 128                          
    coor1 = self.gcn_coor_1_3(coor1, adj)    # 128 → 128 (stage-1 coord feature)  

    nor1  = self.gcn_nor_1_1(nor, adj)       # 64 → 128 on normal stream (stage-1 normal) 

    # Cross-stream fusion/selection (Stage 1)
    coor_nor1 = self.aff_1(coor1, nor1)      # AFF(128): fuse coord/normal @ stage 1      

    # ------------------------------------------------------------
    # Stage 2  →  deeper GCNs + fusion
    # ------------------------------------------------------------
    coor2_1 = self.gcn_coor_2_1(coor1, adj)  # 128 → 256                                   
    coor2_2 = self.gcn_coor_2_2(coor1, adj)  # 128 → 256 (parallel branch)                 
    coor2   = self.aff_coor_2(coor2_1, coor2_2)  # fuse the two coord branches (256)       

    nor2    = self.gcn_nor_2_1(nor1, adj)    # 128 → 256 on normal stream                  

    coor_nor2 = self.aff_2(coor2, nor2)      # AFF(256): fuse coord/normal @ stage 2       

    # ------------------------------------------------------------
    # Stage 3  →  deeper GCNs + fusion
    # ------------------------------------------------------------
    coor3_1 = self.gcn_coor_3_1(coor2, adj)  # 256 → 512                                   
    coor3_2 = self.gcn_coor_3_2(coor2, adj)  # 256 → 512 (parallel)                        
    coor3   = self.aff_coor_3(coor3_1, coor3_2)  # fuse coord branches (512)               

    nor3    = self.gcn_nor_3_1(nor2, adj)    # 256 → 512 on normal stream                  

    coor_nor3 = self.aff_3(coor3, nor3)      # AFF(512): fuse coord/normal @ stage 3       

    # ------------------------------------------------------------
    # Stage 4  →  deeper GCNs + fusion
    # ------------------------------------------------------------
    coor4_1 = self.gcn_coor_4_1(coor3, adj)  # 512 → 512                                   
    coor4_2 = self.gcn_coor_4_2(coor3, adj)  # 512 → 512 (parallel)                        
    coor4   = self.aff_coor_4(coor4_1, coor4_2)  # fuse coord branches (512)               

    nor4    = self.gcn_nor_4_1(nor3, adj)    # 512 → 512 on normal stream                  

    coor_nor4 = self.aff_4(coor4, nor4)      # AFF(512): fuse coord/normal @ stage 4       

    # ------------------------------------------------------------
    # Hierarchical fusion  →  "F_union" style aggregations
    # (Concatenate early + mid levels, compress; then late levels, compress)
    # ------------------------------------------------------------
    x1 = torch.cat((coor_nor1, coor_nor2), dim=1)  # [128 || 256] → 384 ch                 
    x1 = self.fu_1(x1)                             # 384 → 512                             

    x2 = torch.cat((coor_nor3, coor_nor4), dim=1)  # [512 || 512] → 1024 ch                
    x2 = self.fu_2(x2)                             # 1024 → 512                            

    x  = torch.cat((x1, x2), dim=1)                # 512 || 512 → 1024 ch                  
    x  = self.fa_1(x).transpose(-1, -2)            # 1×1 conv attn on channels; to [B,N,1024]

    # ------------------------------------------------------------
    # Segmentation head  →  per-vertex classification
    # ------------------------------------------------------------
    x     = self.pred1(x)                          # 1024 → 512                              
    x     = self.pred2(x)                          # 512  → 256                              
    x     = self.pred3(x)                          # 256  → 128                              
    score = self.pred4(x)                          # 128  → output_channels (e.g., 8)       
    score = F.log_softmax(score, dim=2)            # [B, N, num_classes]                    
    return score