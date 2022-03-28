import torch.nn as nn
from UMRFormer_Net.network_architecture.IntmdSequential import IntermediateSequential


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        # return self.fn(self.norm(x))
        return self.norm(self.fn(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        # return self.dropout(self.fn(self.norm(x)))
        return self.dropout(self.norm(self.fn(x)))


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()

        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        # x = self.pointwise(x)
        return x



class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, input_resolution=None, out_features=None, act_layer=nn.GELU, dropout_rate=0.1):
        super().__init__()

        self.input_resolution = input_resolution
        out_features = out_features or in_dim
        hidden_features = in_dim
        self.layer_norm = nn.LayerNorm(in_dim)

        self.fc1 = nn.Linear(in_dim, hidden_features)

        self.depth_wise_conv1 = SeparableConv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                                dilation=1, bias=False)
        self.layer_norm1 = nn.LayerNorm(hidden_features)
        self.act1 = act_layer()

        self.depth_wise_conv2 = SeparableConv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                                dilation=1, bias=False)
        self.layer_norm2 = nn.LayerNorm(hidden_features)
        self.act2 = act_layer()

        self.depth_wise_conv3 = SeparableConv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                                dilation=1, bias=False)
        self.layer_norm3 = nn.LayerNorm(hidden_features)
        self.act3 = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):

        _, D, H, W, _ = self.input_resolution   # ([2, 512, 8, 16, 16])
        B, L, C = x.shape
        assert L == H * W * D, "input feature has wrong size"

        x = self.fc1(x)

        shortcut = x
        x1 = x.view(B, D, H, W, C)
        x1 = x1.permute(0, 4, 1, 2, 3).contiguous()
        x1 = self.depth_wise_conv1(x1)
        x1 = self.act1(x1)
        x1 = x1.permute(0, 2, 3, 4, 1).contiguous()
        x1 = x1.view(B, H * W * D, C)
        x1 = self.layer_norm1(x1)
        x1 = shortcut + x1

        x2 = x1.view(B, D, H, W, C)
        x2 = x2.permute(0, 4, 1, 2, 3).contiguous()
        x2 = self.depth_wise_conv2(x2)
        x2 = self.act2(x2)
        x2 = x2.permute(0, 2, 3, 4, 1).contiguous()
        x2 = x2.view(B, H * W * D, C)
        x2 = self.layer_norm2(x2)
        x2 = shortcut + x1 + x2

        x3 = x2.view(B, D, H, W, C)
        x3 = x3.permute(0, 4, 1, 2, 3).contiguous()
        x3 = self.depth_wise_conv3(x3)
        x3 = self.act3(x3)
        x3 = x3.permute(0, 2, 3, 4, 1).contiguous()
        x3 = x3.view(B, H * W * D, C)
        x3 = self.layer_norm3(x3)
        x3 = shortcut + x1 + x2 + x3

        x4 = self.fc2(x3)
        x4 = self.drop(x4)
        return x4


class MRFormer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        input_resolution,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, input_resolution=input_resolution, out_features=None, act_layer=nn.GELU, dropout_rate=dropout_rate))
                    ),
                ]
            )
        self.net = IntermediateSequential(*layers)


    def forward(self, x):

        return self.net(x)
