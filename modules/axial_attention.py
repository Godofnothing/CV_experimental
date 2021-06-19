import torch
import torch.nn as nn

from .attention import ResidualAttentionBlock


class AxialEncoderBlock(nn.Module):

    def __init__(
        self, 
        embed_dim: int, 
        feedforward_dim: int,
        n_heads: int,
        dropout: float=0.0,
        activation: str = 'GELU'
    ):
        super().__init__()
        self.x_attention = ResidualAttentionBlock(embed_dim, feedforward_dim, n_heads, dropout=dropout, activation=activation)
        self.y_attention = ResidualAttentionBlock(embed_dim, feedforward_dim, n_heads, dropout=dropout, activation=activation)

    def forward(self, x: torch.Tensor):
        '''
        args:
            x : torch.Tensor (n_x, n_y, batch_size, embed_dim)
        '''
        n_x, n_y = x.shape[:2]

        x = torch.stack(
            [self.x_attention(x[i_x, ...]) for i_x in range(n_x)], dim=1
        )

        x = torch.stack(
            [self.y_attention(x[:, i_y, ...]) for i_y in range(n_y)], dim=1
        ).permute(1, 0, 2, 3)

        return x