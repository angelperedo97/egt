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
        self.use_global_attention = False  # Toggle for global vs. local attention
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            
    def propagate_attention(self, g, use_global_attention=False):
        if use_global_attention:
            # Global attention (fully connected): Compute attention scores for all node pairs
            Q_h = g.ndata['Q_h']  # Shape: [num_nodes, num_heads, out_dim]
            K_h = g.ndata['K_h']
            V_h = g.ndata['V_h']

            # Compute attention scores (scaled dot-product)
            all_pairs_score = torch.matmul(Q_h, K_h.transpose(-2, -1)) / np.sqrt(self.out_dim)
            all_pairs_score = torch.exp(all_pairs_score.clamp(-5, 5))  # Apply exponential and clamp for stability

            # Apply adjacency mask (only consider connected nodes)
            adjacency_matrix = g.adj().to_dense()  # Shape: [num_nodes, num_nodes]
            adjacency_mask = adjacency_matrix.unsqueeze(1).repeat(1, self.num_heads, 1)  # Match head dimensions
            all_pairs_score = all_pairs_score * (adjacency_mask + 1)  # Add 1 to avoid zeroing out

            # Apply softmax normalization on the scores
            all_pairs_score = F.softmax(all_pairs_score, dim=-1)

            # Compute weighted values for each node
            h_out = torch.matmul(all_pairs_score, V_h)  # Shape: [num_nodes, num_heads, out_dim]
            g.ndata['wV'] = h_out  # Store for normalization step
            g.ndata['z'] = all_pairs_score.sum(dim=-1)  # Sum of attention scores per node
        else:
            # Local attention: Calculate attention only on edges

            # Compute attention scores on edges
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
                
    def forward(self, g, h, use_global_attention=False):
        self.use_global_attention = use_global_attention

        # Compute Q, K, V projections
        Q_h = self.Q(h)  # [num_nodes, num_heads * out_dim]
        K_h = self.K(h)
        V_h = self.V(h)

        # Store in node data
        g.ndata['Q_h'] = Q_h
        g.ndata['K_h'] = K_h
        g.ndata['V_h'] = V_h

        # Initiate the attention propagation
        self.propagate_attention(g, use_global_attention)

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

        self.in_channels = in_dim # Input dimension of the node features
        self.out_channels = out_dim # Output dimension of the node features
        self.num_heads = num_heads # Number of attention heads, allowing the model to focus on different aspects of node relationships
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        # Linear transformations applied after attention and aggregation, helping to project the outputs back into the original feature space (out_dim)
        self.O_h = nn.Linear(out_dim, out_dim) 
        
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            
            
    def forward(self, g, h, use_global_attention=False):
        h_in1 = h  # for first residual connection

        # Multi-head attention with option for local/global mode
        h_attn_out = self.attention(g, h, use_global_attention=use_global_attention)

        # Reshape and project the output of the attention
        h = h_attn_out.view(-1, self.out_channels)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O_h(h)

        # Residual connection
        if self.residual:
            h = h_in1 + h

        # Optional layer normalization
        if self.layer_norm:
            h = self.layer_norm1_h(h)

        # Optional batch normalization
        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection in FFN

        # Feed-forward network
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # Second residual connection
        if self.residual:
            h = h_in2 + h

        # Optional second layer normalization
        if self.layer_norm:
            h = self.layer_norm2_h(h)

        # Optional second batch normalization
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

        # Node feature embedding
        self.embedding_h = nn.Embedding(net_params['num_atom_type'], hidden_dim)

        # Dropout on input features
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # Stacked graph transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                  self.layer_norm, self.batch_norm, self.residual)
            for _ in range(n_layers)
        ])

        # Output projection layer to map to desired output dimension
        self.out_proj = nn.Linear(hidden_dim, 1)  # For binary classification

    def forward(self, g, h, use_global_attention=False):
        h = self.embedding_h(h).to(h.device)
        h = self.in_feat_dropout(h)

        for layer in self.layers:
            h = layer(g, h, use_global_attention=use_global_attention)

        g.ndata['h'] = h
        graph_embedding = dgl.mean_nodes(g, 'h')

        output = self.out_proj(graph_embedding)
        return output.squeeze()

    def loss(self, scores, targets):
        loss = nn.BCEWithLogitsLoss()(scores, targets)
        return loss

