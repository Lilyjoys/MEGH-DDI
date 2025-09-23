import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import (SAGPooling,LayerNorm,global_add_pool,Linear,global_mean_pool,GATConv)
from layer import (Hop2TokenEncoder,TransformerEncoderReadout,GCN2Edge,CoAttentionLayer,RESCAL,Graph_intra,Graph_inter,GlobalLayer)


class HOPGCNII_DDI(nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidden_dim, kge_dim, rel_total, heads_out_feat_params,
                 blocks_params, edge_feature, dp):
        super().__init__()
        self.in_node_features = in_node_features[0]
        self.in_node_features_mol2vec = in_node_features[1]
        self.in_node_features_molT5 = in_node_features[2]
        self.in_edge_features = in_edge_features
        self.hidden_dim = hidden_dim #128
        self.kge_dim = kge_dim
        self.rel_total = rel_total
        self.n_blocks = len(blocks_params)  #4
        self.GlobalLayer_feature = GlobalLayer(
            in_features_mol2vec=self.in_node_features_mol2vec,
            in_features_molT5=self.in_node_features_molT5,
            kge_dim=self.kge_dim
        )

        self.lin_node = Linear(self.in_node_features, self.hidden_dim, bias=True, weight_initializer='glorot')
        self.lin_edge = Linear(self.in_edge_features, edge_feature, bias=True, weight_initializer='glorot')
        self.norm_node = LayerNorm(self.hidden_dim)
        self.blocks = []
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = HOPGCNII_DDI_Block(self.hidden_dim, n_heads, head_out_feats, edge_feature, dp)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels, b_graph, h_data_edge, t_data_edge,h_mol2vec,t_mol2vec,h_molT5,t_molT5 = triples
        h_data.x = self.lin_node(h_data.x)
        t_data.x = self.lin_node(t_data.x)
        h_data.x = self.norm_node(h_data.x, h_data.batch)
        t_data.x = self.norm_node(t_data.x, t_data.batch)
        h_data.x = F.relu(h_data.x)
        t_data.x = F.relu(t_data.x)

        h_data.edge_attr = self.lin_edge(h_data.edge_attr)
        t_data.edge_attr = self.lin_edge(t_data.edge_attr)
        h_data.edge_attr = F.relu(h_data.edge_attr)
        t_data.edge_attr = F.relu(t_data.edge_attr)
        h_graphs = []
        t_graphs = []
        for i, block in enumerate(self.blocks):
            data_feature = block(h_data, t_data, b_graph, h_data_edge, t_data_edge)
            h_data = data_feature[0]
            t_data = data_feature[1]
            h_graph = data_feature[2]
            t_graph = data_feature[3]
            h_graphs.append(h_graph)
            t_graphs.append(t_graph)
        _, _, h_data_mol2vec, h_data_molT5, t_data_mol2vec, t_data_molT5 = self.GlobalLayer_feature(h_mol2vec,h_molT5,t_mol2vec,t_molT5)
        h_feaures = []
        t_feaures = []
        for i in range(len(self.blocks)):
            h_feaures.append(F.normalize(h_graphs[i] + h_data_mol2vec+h_data_molT5))
            t_feaures.append(F.normalize(t_graphs[i] + t_data_mol2vec+t_data_molT5))
        h_graphs = torch.stack((h_feaures), dim=1)
        t_graphs = torch.stack((t_feaures), dim=1)
        attentions = self.co_attention(h_graphs, t_graphs)
        scores = self.KGE(h_graphs, t_graphs, rels, attentions)
        return scores
class HOPGCNII_DDI_Block(nn.Module):
    def __init__(self, in_features, n_heads, head_out_feats, edge_feature, dp):
        super().__init__()
        self.n_heads = n_heads #4
        self.in_features = in_features #128
        self.out_features = head_out_feats#32
        self.GCN_convs = nn.ModuleList([
            GCN2Edge(in_features, head_out_feats, i + 1, edge_dim=edge_feature)
            for i in range(n_heads)
        ])
        self.hidden_dim = 128
        self.GAT_conv = GATConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature, dropout=dp)
        self.graph_intras = Graph_intra(head_out_feats * n_heads, dp, n_heads, edge_feature, head_out_feats)
        self.graph_inters = Graph_inter(head_out_feats * n_heads, dp, n_heads, edge_feature, head_out_feats)
        self.pooling = SAGPooling(n_heads * head_out_feats, min_score=-1)
        self.norm = LayerNorm(n_heads * head_out_feats)

        self.lin_up = Linear(edge_feature, edge_feature, bias=True, weight_initializer='glorot')

        self.lin_edge = Linear(edge_feature, n_heads * head_out_feats, bias=True, weight_initializer='glorot')
        self.hop2token = Hop2TokenEncoder(max_hop=3)
        self.token_transformer = TransformerEncoderReadout(
            in_dim=n_heads * head_out_feats,  # 128
            num_layers=2,
            num_heads=4,
            hidden_dim=128,
            dropout=0.1
        )


    def forward(self, h_data, t_data, b_graph, h_data_edge, t_data_edge):
        #GATConv
        h_data.x = self.GAT_conv(h_data.x, h_data.edge_index, h_data.edge_attr)
        t_data.x = self.GAT_conv(t_data.x, t_data.edge_index, t_data.edge_attr)
        #GCNII
        x_h = h_data.x
        x0_h = x_h
        head_outs_h = [GCN_conv(x_h, x0_h, h_data.edge_index, h_data.edge_attr) for GCN_conv in self.GCN_convs]
        x_h = torch.cat(head_outs_h, dim=-1)
        h_gcn = global_add_pool(x_h, h_data.batch)  # [B, hidden]

        x_t = t_data.x
        x0_t = x_t
        head_outs_t = [GCN_conv(x_t, x0_t, t_data.edge_index, t_data.edge_attr) for GCN_conv in self.GCN_convs]
        x_t = torch.cat(head_outs_t, dim=-1)
        t_gcn = global_add_pool(x_t, t_data.batch)  # [B, hidden]

        #Hop2token
        h_token_seq = self.hop2token(h_data.x, h_data.edge_index, h_data.x.size(0))  # [N₁, K+1, 128]
        t_token_seq = self.hop2token(t_data.x, t_data.edge_index, t_data.x.size(0))  # [N₂, K+1, 128]

        # Transformer
        h_trans = self.token_transformer(h_token_seq)  # [N₁, 128]
        t_trans = self.token_transformer(t_token_seq)  # [N₂, 128]

        h_tran = global_add_pool(h_trans, h_data.batch)  # [B, 128]
        t_tran = global_add_pool(t_trans, t_data.batch)  # [B, 128]

        h_gcn_tran = h_gcn + h_tran
        t_gcn_tran = t_gcn + t_tran

        h_data.x = self.norm(h_data.x, h_data.batch)
        t_data.x = self.norm(t_data.x, t_data.batch)
        h_data.edge_attr = self.lin_up(h_data.edge_attr)
        t_data.edge_attr = self.lin_up(t_data.edge_attr)
        #graph inter/intra
        h_intra_feature = self.graph_intras(h_data)
        t_intra_feature = self.graph_intras(t_data)
        h_inter_feature, t_inter_feature = self.graph_inters(h_data, t_data, b_graph)# [N, 128],[N, 128] #[Nt, 64]
        h_intra_inter = torch.cat([h_intra_feature, h_inter_feature], 1) # [N, 64] + [N, 64] = [N, 128]
        t_intra_inter = torch.cat([t_intra_feature, t_inter_feature], 1)# [N, 64] + [N, 64] = [N, 128]
        h_data.x = h_intra_inter #[N, 128]
        t_data.x = t_intra_inter#[N, 128]
        #pooling
        h_x_pooling, edge_index_pooling,edge_attr_pooling, h_batch_pooling, h_perm_pooling, h_scores_pooling = (
            self.pooling(h_data.x,h_data.edge_index,batch=h_data.batch))
        t_x_pooling, edge_index_pooling,edge_attr_pooling, t_batch_pooling, t_perm_pooling, t_scores_pooling = (
            self.pooling(t_data.x, t_data.edge_index,batch=t_data.batch))

        h_graph_gat = global_add_pool(h_x_pooling,h_batch_pooling)  #[N, 128]
        t_graph_gat = global_add_pool(t_x_pooling,t_batch_pooling)

        h_data_edge.x = h_data.edge_attr #64
        t_data_edge.x = t_data.edge_attr
        h_edge_feature = global_add_pool(h_data_edge.x, batch=h_data_edge.batch)
        t_edge_feature = global_add_pool(t_data_edge.x, batch=t_data_edge.batch)

        h_edge = F.relu(self.lin_edge(h_edge_feature))
        t_edge = F.relu(self.lin_edge(t_edge_feature))

        h_graph = h_graph_gat+h_edge+h_gcn_tran #[B₁, 128]
        t_graph = t_graph_gat+t_edge+t_gcn_tran#[B₁, 128]

        h_data.x = F.relu(self.norm(h_data.x, h_data.batch))  # 🔢 [N₁, 128]
        t_data.x = F.relu(self.norm(t_data.x, t_data.batch))   # 🔢 [N₂, 128]
        h_data.edge_attr = F.relu(h_data.edge_attr) # 🔢 [E₁, 64]
        t_data.edge_attr = F.relu(t_data.edge_attr) # 🔢 [E₂, 64]

        return h_data, t_data, h_graph, t_graph




