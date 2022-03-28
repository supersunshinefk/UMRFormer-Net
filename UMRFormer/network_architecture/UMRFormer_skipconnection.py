import torch
import torch.nn as nn
from einops import rearrange
from copy import deepcopy
from UMRFormer_Net.utilities.nd_softmax import softmax_helper
import torch.nn.functional as F
from UMRFormer_Net.network_architecture.Transformer import MRFormer
from UMRFormer_Net.network_architecture.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from UMRFormer_Net.network_architecture.Unet_skipconnection import Unet
from UMRFormer_Net.network_architecture.initialization import InitWeights_He
from UMRFormer_Net.network_architecture.neural_network import SegmentationNetwork


class UTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        input_resolution,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(UTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.input_resolution = [
            input_resolution[0],
            int(input_resolution[2] / patch_dim),
            int(input_resolution[3] / patch_dim),
            int(input_resolution[4] / patch_dim),
            embedding_dim,
        ]
        self.input_resolution_x4_1 = [
            input_resolution[0],
            int((input_resolution[2] / patch_dim))*2,
            int((input_resolution[3] / patch_dim))*2,
            int((input_resolution[4] / patch_dim))*2,
            512,
        ]

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)

        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length,bool_x=True
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.position_encoding_x4_1 = LearnedPositionalEncoding(
            self.seq_length, 512, self.seq_length, bool_x=False
        )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = MRFormer(
            self.embedding_dim,
            num_layers, # 4 layer  transformer block
            num_heads,
            hidden_dim,
            self.input_resolution,
            self.dropout_rate,
            self.attn_dropout_rate,
        )

        self.transformer_x4_1 = MRFormer(
            512,
            8,  # 8 layer  transformer block
            num_heads,
            4096,
            self.input_resolution_x4_1,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)
        self.pre_head_ln_x4_1 = nn.LayerNorm(512)

        if self.conv_patch_representation:

            self.conv_x = nn.Conv3d(
                256,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )
            self.conv_x4_1 = nn.Conv3d(
                128,
                self.embedding_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1
            )


        self.Unet = Unet(in_channels=1, base_channels=16, num_classes=14)
        self.bn = nn.BatchNorm3d(256)
        self.bn_x4_1 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)

    def encode(self, x):
        _,_,self.L,self.H,self.W = x.shape
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution

            x1_1, x2_1, x3_1, x4_1, x = self.Unet(x)

            x = self.bn(x)
            x = self.relu(x)
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)

            x4_1 = self.bn_x4_1(x4_1)
            x4_1 = self.relu(x4_1)
            x4_1 = self.conv_x4_1(x4_1)
            x4_1 = x4_1.permute(0, 2, 3, 4, 1).contiguous()
            x4_1 = x4_1.view(x4_1.size(0), -1, 512)
            x4_1 = self.position_encoding_x4_1(x4_1)
            x4_1 = self.pe_dropout(x4_1)
            x4_1, intmd_x4_1 = self.transformer_x4_1(x4_1)
            x4_1 = self.pre_head_ln_x4_1(x4_1)

            x4_1 = x4_1.view(
                x4_1.size(0),
                int((self.L / self.patch_dim)*2),
                int((self.H / self.patch_dim)*2),
                int((self.W / self.patch_dim)*2),
                512,
            )
            x4_1 = x4_1.permute(0, 4, 1, 2, 3).contiguous()

        else:
            x = self.Unet(x)
            x = self.bn(x)
            x = self.relu(x)
            x = (
                x.unfold(2, 2, 2)
                .unfold(3, 2, 2)
                .unfold(4, 2, 2)
                .contiguous()
            )
            x = x.view(x.size(0), x.size(1), -1, 8)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        return x1_1, x2_1, x3_1, x4_1, x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x, auxillary_output_layers=[1, 2,3,4]):

        x1_1, x2_1, x3_1, x4_1, encoder_output, intmd_encoder_outputs = self.encode(x)

        decoder_output = self.decode(
            x1_1, x2_1, x3_1, x4_1, encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output

        return decoder_output

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.L / self.patch_dim),
            int(self.H / self.patch_dim),
            int(self.W / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class Former(UTransformer):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        input_resolution,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(BTS, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            input_resolution=input_resolution,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes

        self.Softmax = nn.Softmax(dim=1)

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        self.Enblock8_3 = EnBlock3(in_channels=512)
        self.Enblock8_4 = EnBlock4(in_channels=512 // 4)

        # ###########
        self.endconv6 = nn.Conv3d(self.embedding_dim // 4, self.num_classes, kernel_size=1)

        self.DeUp5 = DeUp_Cat(in_channels=self.embedding_dim // 4, out_channels=self.embedding_dim // 8)
        self.DeBlock5 = DeBlock(in_channels=self.embedding_dim // 8)
        # ###########
        self.endconv5 = nn.Conv3d(self.embedding_dim // 8, self.num_classes, kernel_size=1)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim//8, out_channels=self.embedding_dim//16)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim//16)
        # ###########
        self.endconv4 = nn.Conv3d(self.embedding_dim // 16, self.num_classes, kernel_size=1)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim//16, out_channels=self.embedding_dim//32)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim//32)
        # ###########
        self.endconv3 = nn.Conv3d(self.embedding_dim // 32, self.num_classes, kernel_size=1)

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim//32, out_channels=self.embedding_dim//64)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim//64)
        # ###########
        self.endconv2 = nn.Conv3d(self.embedding_dim // 64, self.num_classes, kernel_size=1)




    def decode(self, x1_1, x2_1, x3_1, x4_1, x, intmd_x, intmd_layers=[1, 2, 3, 4]):

        assert intmd_layers is not None, "pass the intermediate layers for MLA"
        encoder_outputs = {}
        all_keys = []
        outs = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            # val = i - 1
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        x8 = encoder_outputs[all_keys[0]]
        x8 = self._reshape_output(x8)
        x8 = self.Enblock8_1(x8)
        x8 = self.Enblock8_2(x8)
        x8_1 = self.endconv6(x8)
        x8_1_B,x8_1_C, x8_1_D, x8_1_H,x8_1_W= x8_1.shape
        # x8_1 = F.interpolate(x8_1, size=(x8_1_H,x8_1_H,x8_1_H), mode="trilinear", align_corners=True)

        outs.append(x8_1)

        x4_1 = self.Enblock8_3(x4_1)
        x4_1 = self.Enblock8_4(x4_1)
        y5 = self.DeUp5(x8, x4_1)  # (1, 128, 16, 16, 16)
        y5 = self.DeBlock5(y5)
        y5_1 = self.endconv5(y5)

        # MSD and Sypanse datasets need upsample
        # output >torch.Size([2, 2, 64, 128, 128]) torch.Size([2, 2, 64, 64, 64]) torch.Size([2, 2, 32, 32, 32])
        # torch.Size([2, 2, 16, 16, 16]) torch.Size([2, 2, 8, 8, 8])
        # target >torch.Size([2, 1, 64, 128, 128]) torch.Size([2, 1, 64, 64, 64]) torch.Size([2, 1, 32, 32, 32])
        # torch.Size([2, 1, 16, 16, 16]) torch.Size([2, 1, 8, 8, 8])
        y5_1_B,y5_1_C, y5_1_D, y5_1_H,y5_1_W= y5_1.shape
        # y5_1 = F.interpolate(y5_1, size=(y5_1_H,y5_1_H,y5_1_H), mode="trilinear", align_corners=True)

        # ZJPancreas/ZJPancreasCancer datasets  not need upsample
        # output >torch.Size([2, 2, 64, 128, 128]) torch.Size([2, 2, 64, 64, 64]) torch.Size([2, 2, 32, 32, 32])
        # torch.Size([2, 2, 16, 16, 16]) torch.Size([2, 2, 8, 8, 8])
        # target >torch.Size([2, 1, 64, 128, 128]) torch.Size([2, 1, 32, 64, 64]) torch.Size([2, 1, 16, 32, 32])
        # torch.Size([2, 1, 8, 16, 16]) torch.Size([2, 1, 4, 8, 8])
        outs.append(y5_1)

        y4 = self.DeUp4(y5, x3_1)  # (1, 64, 32, 32, 32)
        y4 = self.DeBlock4(y4)
        y4_1 = self.endconv4(y4)

        y4_1_B,y4_1_C, y4_1_D, y4_1_H,y4_1_W= y4_1.shape
        # y4_1 = F.interpolate(y4_1, size=(y4_1_H,y4_1_H,y4_1_H), mode="trilinear", align_corners=True)
        outs.append(y4_1)
        y3 = self.DeUp3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)
        y3_1 = self.endconv3(y3)


        y3_1_B,y3_1_C, y3_1_D, y3_1_H,y3_1_W= y3_1.shape
        # y3_1 = F.interpolate(y3_1, size=(y3_1_H,y3_1_H,y3_1_H), mode="trilinear", align_corners=True)
        outs.append(y3_1)
        y2 = self.DeUp2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)
        y = self.endconv2(y2)      # (1, 4, 128, 128, 128)
        # y4_1_B,y4_1_C, y4_1_D, y4_1_H,y4_1_W= y4_1.shape
        # y4_1 = F.interpolate(y4_1, size=(y4_1_H,y4_1_H,y4_1_H))
        outs.append(y)
        return outs

class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm3d(1024 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(1024 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(1024 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(1024 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1

class EnBlock3(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock3, self).__init__()

        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock4(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock4, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1



class UMRFormer(SegmentationNetwork):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=None,
                 seg_output_use_bias=False):
        super(UMRFormer, self).__init__()
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op

        self.upscale_logits_ops = []
        # for usl in range(num_pool - 1):
        # for usl in range(4):
        self.upscale_logits_ops.append(lambda x: x)

        dataset = 'brats'
        _conv_repr = True
        _pe_type = "learned"
        if dataset.lower() == 'brats':
            img_dim = 128

            # Sypanse datasets 14 classes
            # num_classes = 14
            # MSD datasets 3 classes
            # num_classes = 3
            # ZJPancreas/ZJPancreasCancer
            num_classes = 3

        self.num_layers = 4
        # self.num_layers = 2
        num_channels = 1
        # patch_dim = 8
        patch_dim = 16
        aux_layers = [1, 2, 3, 4]
        self.input_resolution = [2,1,64,128,128]
        self.model = Former(
            img_dim,
            patch_dim,
            num_channels,
            num_classes,
            input_resolution = self.input_resolution,
            # embedding_dim=512,
            embedding_dim=1024,
            num_heads=8,
            num_layers=self.num_layers,
            # hidden_dim=4096,
            hidden_dim=8192,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            conv_patch_representation=_conv_repr,
            positional_encoding_type=_pe_type,
        )

    def forward(self, x):

        seg_outputs = []
        outs = self.model(x)
        for i in range(len(outs)):
            seg_outputs.append(self.final_nonlin(outs[(i)]))
        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:

            return seg_outputs[-1]
