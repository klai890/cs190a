import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNRNNModel(nn.Module):
    def __init__(self, in_channels, gnn_hidden_dim, rnn_hidden_dim, out_channels, num_layers=1):
        super(GNNRNNModel, self).__init__()

        # GNN to encode spatial structure at each time step
        self.gnn = GCNConv(in_channels, gnn_hidden_dim)

        # RNN to capture temporal patterns (sequence over time)
        self.rnn = nn.LSTM(
            input_size=gnn_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Final prediction layer (maps RNN output to forecast)
        self.fc = nn.Linear(rnn_hidden_dim, out_channels)

    def forward(self, data_sequence):
        """
        Args:
            data_sequence: List of `Data` objects (one per time step).
                           Each Data has `.x`, `.edge_index`, etc.

        Returns:
            Tensor of shape [num_nodes, out_channels] (e.g., demand forecast)
        """
        node_embeddings_over_time = []

        for data in data_sequence:
            x = data.x         # shape: [num_nodes, in_channels]
            edge_index = data.edge_index

            h = self.gnn(x, edge_index)  # shape: [num_nodes, gnn_hidden_dim]
            node_embeddings_over_time.append(h)

        # Stack over time → shape: [batch, seq_len, feature] = [num_nodes, T, gnn_hidden_dim]
        node_embeddings_seq = torch.stack(node_embeddings_over_time, dim=1)

        # Apply RNN across time steps
        rnn_out, _ = self.rnn(node_embeddings_seq)  # shape: [num_nodes, T, rnn_hidden_dim]

        # Get last time step output
        last_hidden = rnn_out[:, -1, :]  # shape: [num_nodes, rnn_hidden_dim]

        # Final prediction (e.g., forecast demand at next time step)
        out = self.fc(last_hidden)  # shape: [num_nodes, out_channels]

        return out
