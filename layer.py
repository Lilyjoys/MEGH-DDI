from get_args import config
from torch_geometric.nn import GATConv
layer_dataset_name = config['dataset_name']
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
def compute_avg_degree(edge_index, batch, num_nodes):

    node_deg = degree(edge_index[0], num_nodes=num_nodes)  # [num_nodes]

    num_graphs = batch.max().item() + 1
    avg_degree_list = []
    for i in range(num_graphs):
        mask = (batch == i)
        avg_deg = node_deg[mask].mean()
        avg_degree_list.append(avg_deg)
    return torch.stack(avg_degree_list).unsqueeze(1)  # [batch_size, 1]
class StructureAwareMultiHeadBiAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, structure_dim=1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.struct_proj = nn.Linear(structure_dim, num_heads)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h1, h2, struct1, struct2):
        """
        h1: [B1, hidden_dim], h2: [B2, hidden_dim]
        struct1, struct2: [B1, 1], [B2, 1]
        """
        B1 = h1.size(0)
        B2 = h2.size(0)

        if B1 != B2:
            raise ValueError(f"Mismatch: h1.shape[0]={B1}, h2.shape[0]={B2}. Make sure batch pairs match.")

        q1 = self.q_proj(h1).view(B1, self.num_heads, self.head_dim)
        k2 = self.k_proj(h2).view(B2, self.num_heads, self.head_dim)
        v2 = self.v_proj(h2).view(B2, self.num_heads, self.head_dim)

        struct_weights1 = torch.sigmoid(self.struct_proj(struct1))  # [B1, heads]

        scores1 = (q1 * k2).sum(dim=-1) * self.scale  # [B1, heads]
        scores1 = scores1 * struct_weights1
        alpha1 = torch.softmax(scores1, dim=-1).unsqueeze(-1)

        attended1 = (alpha1 * v2).reshape(B1, self.hidden_dim)
        h1_updated = h1 + self.out_proj(attended1)


        q2 = self.q_proj(h2).view(B2, self.num_heads, self.head_dim)
        k1 = self.k_proj(h1).view(B1, self.num_heads, self.head_dim)
        v1 = self.v_proj(h1).view(B1, self.num_heads, self.head_dim)

        struct_weights2 = torch.sigmoid(self.struct_proj(struct2))  # [B2, heads]
        scores2 = (q2 * k1).sum(dim=-1) * self.scale
        scores2 = scores2 * struct_weights2
        alpha2 = torch.softmax(scores2, dim=-1).unsqueeze(-1)

        attended2 = (alpha2 * v1).reshape(B2, self.hidden_dim)
        h2_updated = h2 + self.out_proj(attended2)

        return h1_updated, h2_updated
class Hop2TokenEncoder(nn.Module):
    def __init__(self, max_hop=3):
        super().__init__()
        self.max_hop = max_hop

    def forward(self, x, edge_index, num_nodes):

        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1), device=x.device),
            (num_nodes, num_nodes)
        )
        x_list = [x]
        Ax = x
        for _ in range(self.max_hop):
            Ax = torch.sparse.mm(adj, Ax)
            x_list.append(Ax)
        token_sequence = torch.stack(x_list, dim=1)  # [N, K+1, D]
        return token_sequence
class TransformerEncoderReadout(nn.Module):
    def __init__(self, in_dim, num_layers=2, num_heads=4, hidden_dim=128, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Linear(in_dim, 1)  # Attention-based readout

    def forward(self, token_sequence):
        """
        token_sequence: [N, K, D] — N nodes, each node has K hop tokens, and each token is represented by a D-dimensional vector.
        """
        # Transformer
        x = self.transformer_encoder(token_sequence)  #  [N, K, D]
        attn_scores = self.attention_pool(x).squeeze(-1)  # [N, K]
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [N, K, 1]
        out = torch.sum(attn_weights * x, dim=1)  # [N, D]
        return out
def add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes, edge_attr_dim):

    edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    num_old_edges = edge_index.size(1)
    num_new_edges = edge_index_with_loops.size(1)
    num_self_loops = num_new_edges - num_old_edges
    if num_self_loops > 0:

        loop_attr = torch.zeros((num_self_loops, edge_attr_dim), device=edge_attr.device, dtype=edge_attr.dtype)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
    return edge_index_with_loops, edge_attr
class EdgeAwareGCN2Conv(torch.nn.Module):
    def __init__(self, channels, alpha, theta, layer, shared_weights=True, edge_dim=64):
        super().__init__()
        self.channels = channels
        self.alpha = alpha
        self.theta = theta
        self.layer = layer
        self.shared_weights = shared_weights
        self.edge_att = nn.Linear(edge_dim, 1)
        self.lin_l = nn.Linear(channels, channels, bias=True)
        self.lin_r = nn.Linear(channels, channels, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_att.weight)
        if self.edge_att.bias is not None:
            nn.init.zeros_(self.edge_att.bias)
        nn.init.xavier_uniform_(self.lin_l.weight)
        if self.lin_l.bias is not None:
            nn.init.zeros_(self.lin_l.bias)
        nn.init.xavier_uniform_(self.lin_r.weight)
        if self.lin_r.bias is not None:
            nn.init.zeros_(self.lin_r.bias)


    def forward(self, x, x0, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops_with_edge_attr(
            edge_index, edge_attr, num_nodes=x.size(0), edge_attr_dim=edge_attr.size(1)
        )
        edge_weight = self.edge_att(edge_attr).squeeze(-1)  # [E]
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # [E]

        out = self.propagate(edge_index, x=x, norm=norm)

        out = (1 - self.alpha) * x + self.alpha * out
        out = (1 - self.theta) * out + self.theta * x0
        return out

    def propagate(self, edge_index, x, norm):
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, norm.view(-1, 1) * x[col])
        return out
class GCN2Edge(nn.Module):
    def __init__(self, in_dim, out_dim, layer_id, edge_dim=64):
        super().__init__()
        self.conv = EdgeAwareGCN2Conv(
            channels=in_dim,
            alpha=0.5,
            theta=0.5,
            layer=layer_id,
            shared_weights=True,
            edge_dim=edge_dim
        )
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, x0, edge_index, edge_attr):
        out = self.conv(x, x0, edge_index, edge_attr)  # [N, in_dim]
        out = self.lin(out)  # [N, out_dim]
        return out

class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features #128
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))  #(128，64)
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))#(128，64)
        self.bias = nn.Parameter(torch.zeros(n_features // 2)) #(64,)
        self.a = nn.Parameter(torch.zeros(n_features // 2))#(64,)

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))


    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k # receiver[B, L, 128]  w_k(128，64) →# [B, L, 64]
        queries = attendant @ self.w_q  #attendant[B, L, 128]  w_q(128，64) →# [B, L, 64]
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias #[B, L, L, 64]
        e_scores = torch.tanh(e_activations) @ self.a #[B, L, L]
        # e_scores = e_activations @ self.a
        attentions = e_scores
        return attentions
class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, rels, alpha_scores):
        if 'drugbank' not in layer_dataset_name:
            heads = F.normalize(heads, dim=-1)
            tails = F.normalize(tails, dim=-1)

            scores = heads @ tails.transpose(-2, -1)
        else:
            rels = self.rel_emb(rels)

            rels = F.normalize(rels, dim=-1)
            heads = F.normalize(heads, dim=-1)
            tails = F.normalize(tails, dim=-1)

            rels = rels.view(-1, self.n_features, self.n_features)

            scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
            scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))

        return scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"
# intra rep
class Graph_intra(nn.Module):
    def __init__(self, input_dim, dp, head, edge, head_out_feats):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim, head_out_feats // 2, head, edge_dim=edge, dropout=dp)

    def forward(self, data):
        input_feature, edge_index = data.x, data.edge_index
        input_feature = F.relu(input_feature)
        intra_rep = self.intra(input_feature, edge_index, data.edge_attr) # [N, 64]
        return intra_rep
# inter rep
class Graph_inter(nn.Module):
    def __init__(self, input_dim, dp, head, edge, head_out_feats):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim, input_dim), head_out_feats // 2, head, dropout=dp)

    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.relu(h_data.x)
        t_input = F.relu(t_data.x)
        t_rep = self.inter((h_input, t_input), edge_index) #[Nt, 64]
        h_rep = self.inter((t_input, h_input), edge_index[[1, 0]]) #[Nt, 64]

        return h_rep, t_rep
class GlobalLayer(nn.Module):
    def __init__(self, in_features_mol2vec, in_features_molT5, kge_dim):
        super().__init__()
        self.in_features_mol2vec = in_features_mol2vec
        self.in_features_molT5 = in_features_molT5
        self.kge_dim = kge_dim
        self.reduction_mol2vec = nn.Sequential(nn.Linear(self.in_features_mol2vec, self.kge_dim),
                                            #   nn.BatchNorm1d(256),
                                            nn.ELU())
        self.reduction_molT5 = nn.Sequential(nn.Linear(self.in_features_molT5, self.kge_dim),
                                            #   nn.BatchNorm1d(256),

                                            nn.ELU(),

                                             )

        self.merge_fd = nn.Sequential(nn.Linear(self.kge_dim * 2, self.kge_dim),
                                      nn.ELU())

    def forward(self, h_data_fin, h_data_desc, t_data_fin, t_data_desc):
        h_data_fin = F.normalize(h_data_fin, 2, 1)
        h_data_desc = F.normalize(h_data_desc, 2, 1)

        t_data_fin = F.normalize(t_data_fin, 2, 1)
        t_data_desc = F.normalize(t_data_desc, 2, 1)

        h_data_fin = self.reduction_mol2vec(h_data_fin)
        h_data_desc = self.reduction_molT5(h_data_desc)

        t_data_fin = self.reduction_mol2vec(t_data_fin)
        t_data_desc = self.reduction_molT5(t_data_desc)

        h_fdmerge = torch.cat((h_data_fin, h_data_desc), dim=1)
        h_fdmerge = F.normalize(h_fdmerge, 2, 1)
        h_fdmerge = self.merge_fd(h_fdmerge)

        t_fdmerge = torch.cat((t_data_fin, t_data_desc), dim=1)
        t_fdmerge = F.normalize(t_fdmerge, 2, 1)
        t_fdmerge = self.merge_fd(t_fdmerge)

        return h_fdmerge, t_fdmerge, h_data_fin, h_data_desc, t_data_fin, t_data_desc



