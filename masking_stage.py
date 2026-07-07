"""
Masking Stage
(U2Net+TLS+TSAIC)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from TSAIC import TSAIC
from tools import ConvSTFT, ConviSTFT
from config import WIN_LEN, HOP_LEN, FFT_LEN

#  convolution
class causalConv2d(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size,
            stride=1,
            padding=(1, 1),  # (pad_freq, pad_time)
            dilation=1,
            groups=1,
            causal=False
    ):
        super().__init__()

        self.causal = causal
        self.pad_freq = padding[0]
        self.pad_time = padding[1]

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=(self.pad_freq, 0),
            dilation=dilation,
            groups=groups,
            bias=False
        )

    def forward(self, x):
        if self.pad_time > 0:
            if self.causal:
                # causal
                x = F.pad(x, [self.pad_time, 0, 0, 0])
            else:
                # non-causal
                left = self.pad_time // 2
                right = self.pad_time - left
                x = F.pad(x, [left, right, 0, 0])

        return self.conv(x)

# convolution block
class CONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CONV, self).__init__()
        self.conv = causalConv2d(in_ch, out_ch, kernel_size=(3, 2), stride=(2, 1), padding=(1, 1))
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.LeakyReLU()

    def forward(self, x):
        return self.prelu(self.ln(self.conv(x)))


# convolution block for input layer
class INCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(INCONV, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.LeakyReLU()

    def forward(self, x):
        return self.prelu(self.ln(self.conv(x)))


# sub-pixel convolution block
class SPCONV(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(SPCONV, self).__init__()
        self.conv = causalConv2d(in_ch, out_ch * scale_factor, kernel_size=(3, 2), padding=(1, 1))
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.LeakyReLU()

        self.n = scale_factor

    def forward(self, x):
        x = self.conv(x)  # [B, C, F, T]

        x = x.permute(0, 3, 2, 1)  # [B, T, F, C]
        r = torch.reshape(x, (x.size(0), x.size(1), x.size(2), x.size(3) // self.n, self.n))  # [B, T, F, C//2 , 2]
        r = r.permute(0, 1, 2, 4, 3)  # [B, T, F, 2, C//2]
        r = torch.reshape(r, (x.size(0), x.size(1), x.size(2) * self.n, x.size(3) // self.n))  # [B, T, F*2, C//2]
        r = r.permute(0, 3, 2, 1)  # [B, C, F, T]

        out = self.ln(r)
        out = self.prelu(out)
        return out


# Convolutional down-sampling along the frequency dimension
class down_sampling(nn.Module):
    def __init__(self, in_ch):
        super(down_sampling, self).__init__()
        self.down_sampling = nn.Conv2d(in_ch, in_ch, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x):
        return self.down_sampling(x)


# Transposed convolution for frequency up-sampling
class upsampling(nn.Module):
    def __init__(self, in_ch):
        super(upsampling, self).__init__()
        self.upsampling = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=(3, 1), stride=(2, 1),
                                             padding=(1, 0), output_padding=(1, 0))

    def forward(self, x):
        out = self.upsampling(x)
        return out


# dilated dense block
class DilatedDenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_layers,
                 dilations=(1, 2, 4, 8, 16,32),
                 kernel_size=(3, 2),
                 use_gate=True):
        super().__init__()

        assert n_layers <= len(dilations)

        self.input_layer = causalConv2d(in_ch, in_ch // 2, kernel_size=kernel_size, padding=(1, 1))
        self.act_in = nn.PReLU()
        self.use_gate = use_gate

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            d = dilations[i]

            pad_f = d
            pad_t = d

            in_c = (in_ch // 2) * (i + 1)  # dense concat channels

            block = [
                causalConv2d(in_c, in_ch // 2, kernel_size=kernel_size,
                             padding=(pad_f, pad_t), dilation=(d, d), groups=in_ch // 2),
                nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1),
                nn.GroupNorm(1, in_ch // 2, eps=1e-8),
            ]

            if use_gate:
                # Gate head
                self.layers.append(nn.ModuleDict({
                    "feat": nn.Sequential(*block, nn.PReLU()),
                    "gate": nn.Sequential(
                        nn.Conv2d(in_c, in_ch // 2, kernel_size=1),
                        nn.Sigmoid()
                    )
                }))
            else:
                self.layers.append(nn.Sequential(*block, nn.PReLU()))

        self.out_layer = causalConv2d(in_ch // 2, out_ch, kernel_size=kernel_size, padding=(1, 1))
        self.act_out = nn.PReLU()

    def forward(self, x):
        x = self.act_in(self.input_layer(x))  # [B, C/2, F, T]

        feats = [x]
        for layer in self.layers:
            inp = torch.cat(feats, dim=1)
            if isinstance(layer, nn.ModuleDict):
                y = layer["feat"](inp)
                g = layer["gate"](inp)
                y = y * g
            else:
                y = layer(inp)
            feats.append(y)

        out = feats[-1]
        out = self.act_out(self.out_layer(out))
        return out



# Variable - Level Features Extraction  ( VLFE) - e6 (for encoder part)
class VLFEe6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEe6, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)
        self.en5 = CONV(mid_ch, mid_ch)
        self.en6 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, mid_ch)
        self.de6 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)
        out5 = self.en5(out4)
        out6 = self.en6(out5)

        # bottleneck
        out = self.ddense(out6)

        # decoder
        out6 = self.de1(torch.cat([out, out6], dim=1))
        out5 = self.de2(torch.cat([out6, out5], dim=1))
        out4 = self.de3(torch.cat([out5, out4], dim=1))
        out3 = self.de4(torch.cat([out4, out3], dim=1))
        out2 = self.de5(torch.cat([out3, out2], dim=1))
        out1 = self.de6(torch.cat([out2, out1], dim=1))

        out = out1 + x
        return out, out1, out2, out3, out4, out5, out6


# Variable - Level Features Extraction  ( VLFE) - e5
class VLFEe5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEe5, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)
        self.en5 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)
        out5 = self.en5(out4)

        # bottleneck
        out = self.ddense(out5)

        # decoder
        out5 = self.de1(torch.cat([out, out5], dim=1))
        out4 = self.de2(torch.cat([out5, out4], dim=1))
        out3 = self.de3(torch.cat([out4, out3], dim=1))
        out2 = self.de4(torch.cat([out3, out2], dim=1))
        out1 = self.de5(torch.cat([out2, out1], dim=1))

        out = out1 + x

        return out, out1, out2, out3, out4, out5


# Variable - Level Features Extraction  ( VLFE) - e4
class VLFEe4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEe4, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)

        # bottleneck
        out = self.ddense(out4)

        # decoder
        out4 = self.de1(torch.cat([out, out4], dim=1))
        out3 = self.de2(torch.cat([out4, out3], dim=1))
        out2 = self.de3(torch.cat([out3, out2], dim=1))
        out1 = self.de4(torch.cat([out2, out1], dim=1))

        out = out1 + x

        return out, out1, out2, out3, out4


# Variable - Level Features Extraction  ( VLFE) - e3
class VLFEe3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEe3, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)

        # bottleneck
        out = self.ddense(out3)

        # decoder
        out3 = self.de1(torch.cat([out, out3], dim=1))
        out2 = self.de2(torch.cat([out3, out2], dim=1))
        out1 = self.de3(torch.cat([out2, out1], dim=1))

        out = out1 + x
        return out, out1, out2, out3


# Variable - Level Features Extraction  ( VLFE) - e2
class VLFEe2(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEe2, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)

        # bottleneck
        out = self.ddense(out2)

        # decoder
        out2 = self.de1(torch.cat([out, out2], dim=1))
        out1 = self.de2(torch.cat([out2, out1], dim=1))

        out = out1 + x
        return out, out1, out2


# Variable - Level Features Extraction  ( VLFE) - e1
class VLFEe1(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEe1, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)

        # bottleneck
        out = self.ddense(out1)

        # decoder
        out1 = self.de1(torch.cat([out, out1], dim=1))

        out = out1 + x
        return out, out1


# Variable - Level Features Extraction  ( VLFE) - d6  (for decoder part)
class VLFEd6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEd6, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch * 2, mid_ch)
        self.en2 = CONV(mid_ch * 2, mid_ch)
        self.en3 = CONV(mid_ch * 2, mid_ch)
        self.en4 = CONV(mid_ch * 2, mid_ch)
        self.en5 = CONV(mid_ch * 2, mid_ch)
        self.en6 = CONV(mid_ch * 2, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, mid_ch)
        self.de6 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x, ed1, ed2, ed3, ed4, ed5, ed6):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(torch.cat([x, ed1], dim=1))
        out2 = self.en2(torch.cat([out1, ed2], dim=1))
        out3 = self.en3(torch.cat([out2, ed3], dim=1))
        out4 = self.en4(torch.cat([out3, ed4], dim=1))
        out5 = self.en5(torch.cat([out4, ed5], dim=1))
        out6 = self.en6(torch.cat([out5, ed6], dim=1))

        # bottleneck
        out = self.ddense(out6)

        # decoder
        out = self.de1(torch.cat([out, out6], dim=1))
        out = self.de2(torch.cat([out, out5], dim=1))
        out = self.de3(torch.cat([out, out4], dim=1))
        out = self.de4(torch.cat([out, out3], dim=1))
        out = self.de5(torch.cat([out, out2], dim=1))
        out = self.de6(torch.cat([out, out1], dim=1))

        out = out + x

        return out


# Variable - Level Features Extraction  ( VLFE) - d5
class VLFEd5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEd5, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch * 2, mid_ch)
        self.en2 = CONV(mid_ch * 2, mid_ch)
        self.en3 = CONV(mid_ch * 2, mid_ch)
        self.en4 = CONV(mid_ch * 2, mid_ch)
        self.en5 = CONV(mid_ch * 2, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)


        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x, ed1, ed2, ed3, ed4, ed5):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(torch.cat([x, ed1], dim=1))
        out2 = self.en2(torch.cat([out1, ed2], dim=1))
        out3 = self.en3(torch.cat([out2, ed3], dim=1))
        out4 = self.en4(torch.cat([out3, ed4], dim=1))
        out5 = self.en5(torch.cat([out4, ed5], dim=1))

        # bottleneck
        out = self.ddense(out5)


        # decoder
        out = self.de1(torch.cat([out, out5], dim=1))
        out = self.de2(torch.cat([out, out4], dim=1))
        out = self.de3(torch.cat([out, out3], dim=1))
        out = self.de4(torch.cat([out, out2], dim=1))
        out = self.de5(torch.cat([out, out1], dim=1))


        out = out + x
        return out


# Variable - Level Features Extraction  ( VLFE) - d4
class VLFEd4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEd4, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch * 2, mid_ch)
        self.en2 = CONV(mid_ch * 2, mid_ch)
        self.en3 = CONV(mid_ch * 2, mid_ch)
        self.en4 = CONV(mid_ch * 2, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x, ed1, ed2, ed3, ed4):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(torch.cat([x, ed1], dim=1))
        out2 = self.en2(torch.cat([out1, ed2], dim=1))
        out3 = self.en3(torch.cat([out2, ed3], dim=1))
        out4 = self.en4(torch.cat([out3, ed4], dim=1))

        # bottleneck
        out = self.ddense(out4)

        # decoder
        out = self.de1(torch.cat([out, out4], dim=1))
        out = self.de2(torch.cat([out, out3], dim=1))
        out = self.de3(torch.cat([out, out2], dim=1))
        out = self.de4(torch.cat([out, out1], dim=1))


        out = out + x

        return out


# Variable - Level Features Extraction  ( VLFE) - d3
class VLFEd3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEd3, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch * 2, mid_ch)
        self.en2 = CONV(mid_ch * 2, mid_ch)
        self.en3 = CONV(mid_ch * 2, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)


        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x, ed1, ed2, ed3):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(torch.cat([x, ed1], dim=1))
        out2 = self.en2(torch.cat([out1, ed2], dim=1))
        out3 = self.en3(torch.cat([out2, ed3], dim=1))

        # bottleneck
        out = self.ddense(out3)


        # decoder
        out = self.de1(torch.cat([out, out3], dim=1))
        out = self.de2(torch.cat([out, out2], dim=1))
        out = self.de3(torch.cat([out, out1], dim=1))

        out = out + x
        return out


# Variable - Level Features Extraction  ( VLFE) - d2
class VLFEd2(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEd2, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch * 2, mid_ch)
        self.en2 = CONV(mid_ch * 2, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x, ed1, ed2):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(torch.cat([x, ed1], dim=1))
        out2 = self.en2(torch.cat([out1, ed2], dim=1))

        # bottleneck
        out = self.ddense(out2)

        # decoder
        out = self.de1(torch.cat([out, out2], dim=1))
        out = self.de2(torch.cat([out, out1], dim=1))

        # attention
        out = out + x
        return out


# Variable - Level Features Extraction  ( VLFE) - d1
class VLFEd1(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super( VLFEd1, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch * 2, mid_ch)

        # bottleneck
        self.ddense = DilatedDenseBlock(mid_ch, mid_ch, 6)


        # decoder
        self.de1 = SPCONV(mid_ch * 2, out_ch)



    def forward(self, x, ed1):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(torch.cat([x, ed1], dim=1))

        # bottleneck
        out = self.ddense(out1)

        # decoder
        out = self.de1(torch.cat([out, out1], dim=1))

        out = out + x

        return out


# Masking stage using U2Net-TLS with TSAIC
class MaskingStage(nn.Module):

    def __init__(self, in_ch=1, mid_ch=32, out_ch=64):
        super(MaskingStage, self).__init__()
        self.fft_len = FFT_LEN

        self.input_layer = INCONV(in_ch, out_ch)

        # ===================== ENCODER =====================
        self.en1 = VLFEe6(out_ch, mid_ch, out_ch)
        self.down_sampling1 = down_sampling(out_ch)

        self.en2 = VLFEe5(out_ch, mid_ch, out_ch)
        self.down_sampling2 = down_sampling(out_ch)

        self.en3 = VLFEe4(out_ch, mid_ch, out_ch)
        self.down_sampling3 = down_sampling(out_ch)

        self.en4 = VLFEe3(out_ch, mid_ch, out_ch)
        self.down_sampling4 = down_sampling(out_ch)

        self.en5 = VLFEe2(out_ch, mid_ch, out_ch)
        self.down_sampling5 = down_sampling(out_ch)

        self.en6 = VLFEe1(out_ch, mid_ch, out_ch)
        self.down_sampling6 = down_sampling(out_ch)

        # ===================== TSAIC BOTTLENECK =====================
        self.tsaic = TSAIC(in_channels=out_ch, hidden_dim=32, alpha=1.0, beta=1.0)

        # ===================== DECODER =====================

        self.upsampling1 = upsampling(out_ch * 2)
        self.de1 = VLFEd1(out_ch * 2, mid_ch, out_ch)

        self.upsampling2 = upsampling(out_ch * 2)
        self.de2 = VLFEd2(out_ch * 2, mid_ch, out_ch)

        self.upsampling3 = upsampling(out_ch * 2)
        self.de3 = VLFEd3(out_ch * 2, mid_ch, out_ch)

        self.upsampling4 = upsampling(out_ch * 2)
        self.de4 = VLFEd4(out_ch * 2, mid_ch, out_ch)

        self.upsampling5 = upsampling(out_ch * 2)
        self.de5 = VLFEd5(out_ch * 2, mid_ch, out_ch)

        self.upsampling6 = upsampling(out_ch * 2)
        self.de6 = VLFEd6(out_ch * 2, mid_ch, out_ch)

        # output layer
        self.output_layer = nn.Conv2d(out_ch, in_ch, kernel_size=1)
        self.mask_sigmoid = nn.Sigmoid()


        # for feature extract
        self.stft = ConvSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='real')
        self.istft = ConviSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='real')

    def forward(self, x):
        # STFT
        mags, phase = self.stft(x)  # [B, F, T]
        hx01 = mags.unsqueeze(1)  # [B, 1, F, T]
        hx02 = hx01[:, :, 1:]

        # input layer
        hx = self.input_layer(hx02)  # [B, out_ch, F, T]

        # ===================== ENCODER PATH =====================
        # encoder stage 1
        hx1, hx1_1, hx1_2, hx1_3, hx1_4, hx1_5, hx1_6 = self.en1(hx)
        hx1 = self.down_sampling1(hx1)

        # encoder stage 2
        hx2, hx2_1, hx2_2, hx2_3, hx2_4, hx2_5 = self.en2(hx1)
        hx2 = self.down_sampling2(hx2)

        # encoder stage 3
        hx3, hx3_1, hx3_2, hx3_3, hx3_4 = self.en3(hx2)
        hx3 = self.down_sampling3(hx3)

        # encoder stage 4
        hx4, hx4_1, hx4_2, hx4_3 = self.en4(hx3)
        hx4 = self.down_sampling4(hx4)

        # encoder stage 5
        hx5, hx5_1, hx5_2 = self.en5(hx4)
        hx5 = self.down_sampling5(hx5)

        # encoder stage 6
        hx6, hx6_1 = self.en6(hx5)
        hx6 = self.down_sampling6(hx6)

        # ===================== TSAIC BOTTLENECK =====================
        out = self.tsaic(hx6)

        # ===================== DECODER PATH =====================

        # decoder stage 1 (Compatible with stage 6)
        out = self.upsampling1(torch.cat([out, hx6], dim=1))
        out = self.de1(out, hx6_1)

        # decoder stage 2 (Compatible with stage 5)
        out = self.upsampling2(torch.cat([out, hx5], dim=1))
        out = self.de2(out, hx5_1, hx5_2)

        # decoder stage 3 (Compatible with stage 4)
        out = self.upsampling3(torch.cat([out, hx4], dim=1))
        out = self.de3(out, hx4_1, hx4_2, hx4_3)

        # decoder stage 4 (Compatible with stage 3)
        out = self.upsampling4(torch.cat([out, hx3], dim=1))
        out = self.de4(out, hx3_1, hx3_2, hx3_3, hx3_4)

        # decoder stage 5 (Compatible with stage 2)
        out = self.upsampling5(torch.cat([out, hx2], dim=1))
        out = self.de5(out, hx2_1, hx2_2, hx2_3, hx2_4, hx2_5)

        # decoder stage 6 (Compatible with stage 1)
        out = self.upsampling6(torch.cat([out, hx1], dim=1))
        out = self.de6(out, hx1_1, hx1_2, hx1_3, hx1_4, hx1_5, hx1_6)

        # Estimate the spectral mask
        out = self.output_layer(out)

        out = self.mask_sigmoid(out)

        # Apply the estimated mask to the noisy magnitude spectrum
        out = out * hx02

        # Restore the DC frequency bin
        out = F.pad(out, [0, 0, 1, 0])

        # Reconstruct the enhanced waveform using the original noisy phase
        out_wav = self.istft(out.squeeze(1), phase).squeeze(1)
        out_wav = torch.clamp_(out_wav, -1, 1)  # Clip the waveform to the range [-1, 1]


        return out_wav

    def loss(self, enhanced, target):

        # Waveform loss
        loss_wave = F.mse_loss(enhanced, target)
        return loss_wave



