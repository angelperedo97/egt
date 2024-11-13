import dgl
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

"""
Utility Functions for attention calculation
"""

# Computes the dot product between source and destination node features for each edge
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

# Scales the attention scores by a constant to stabilize gradients
def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Applies the exponential function to attention scores and clamps them for numerical stability
def exp(field):
    def func(edges):
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            
    def propagate_attention(self, g):
        
        def edge_attention(edges):
            K_h = edges.src['K_h'].view(-1, self.num_heads, self.out_dim)
            Q_h = edges.dst['Q_h'].view(-1, self.num_heads, self.out_dim)
            score = (K_h * Q_h).sum(dim=-1) / np.sqrt(self.out_dim)
            score = torch.exp(score.clamp(-5, 5))
            return {'score': score}

        g.apply_edges(edge_attention)

        def message_func(edges):
            V_h = edges.src['V_h'].view(-1, self.num_heads, self.out_dim)
            score = edges.data['score'].unsqueeze(-1)
            m = V_h * score
            return {'m': m, 'score': edges.data['score']}

        def reduce_func(nodes):
            wV = torch.sum(nodes.mailbox['m'], dim=1)
            z = torch.sum(nodes.mailbox['score'], dim=1).unsqueeze(-1)
            return {'wV': wV, 'z': z}

        # Perform message passing
        g.update_all(message_func, reduce_func)
                
    def forward(self, g, h):
        # Compute Q, K, V projections
        Q_h = self.Q(h)  # [num_nodes, num_heads * out_dim]
        K_h = self.K(h)
        V_h = self.V(h)

        # Store in node data
        g.ndata['Q_h'] = Q_h
        g.ndata['K_h'] = K_h
        g.ndata['V_h'] = V_h

        # Initiate the attention propagation
        self.propagate_attention(g)

        # Normalize the aggregated values (wV) by the sum of the attention scores (z)
        h_out = g.ndata['wV'] / (g.ndata['z'] + 1e-6)
        h_out = h_out.reshape(-1, self.num_heads * self.out_dim)

        # Clean up
        g.ndata.pop('Q_h')
        g.ndata.pop('K_h')
        g.ndata.pop('V_h')
        g.ndata.pop('wV')
        g.ndata.pop('z')
        if 'score' in g.edata:
            g.edata.pop('score')

        return h_out

        
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
        
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            
            
    def forward(self, g, h):
        h_in1 = h

        h_attn_out = self.attention(g, h)

        h = h_attn_out.view(-1, self.out_channels)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h

        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        return h
        
        
class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']

        self.embedding_h = nn.Embedding(net_params['num_atom_type'], hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                  self.layer_norm, self.batch_norm, self.residual)
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, g, h):
        h = self.embedding_h(h).to(h.device)
        h = self.in_feat_dropout(h)

        for layer in self.layers:
            h = layer(g, h)

        g.ndata['h'] = h
        graph_embedding = dgl.mean_nodes(g, 'h')

        output = self.out_proj(graph_embedding)
        return output.squeeze()

    def loss(self, scores, targets):
        return nn.BCEWithLogitsLoss()(scores, targets)

