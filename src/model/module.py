import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Linear(nn.Linear):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            bias: bool = True,
            init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        elif init == "jax":
            self._jax_init()
        elif init == 'constant':
            self._constant_init(0.1)
        else:
            raise ValueError("Invalid init method.")

    def _constant_init(self, constant=1.0):
        nn.init.constant_(self.weight, constant)
        nn.init.constant_(self.bias, 0.0)

    def _trunc_normal_init(self, scale=1.0):
        self.weight = nn.Parameter(torch.randn(self.weight.shape) * 0.1)  # 随机初始化权重
        self.bias = nn.Parameter(torch.zeros(self.bias.shape))  # 初始化偏置为 0
        # # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        # TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        # _, fan_in = self.weight.shape
        # scale = scale / max(1, fan_in)
        # std = (scale ** 0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        # nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def _jax_init(self):
        input_size = self.weight.shape[-1]
        std = math.sqrt(1 / input_size)
        nn.init.trunc_normal_(self.weight, std=std, a=-2.0 * std, b=2.0 * std)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, activation='LeakyReLU', final_init='final', bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.activation = getattr(torch.nn, activation)()
        layers = [Linear(in_channels, hidden_channels, bias), self.activation]
        # import pickle
        # random_data = pickle.load(open('/home/user/tsj/Spectra2Molecule/random_data.pkl', 'rb'))
        # layers[0].weight = nn.Parameter(random_data)
        for _ in range(num_layers):
            layers += [Linear(hidden_channels, hidden_channels, bias), self.activation]
            # random_data_hidden = pickle.load(open('/home/user/tsj/Spectra2Molecule/random_data_hidden.pkl', 'rb'))
            # layers[-1].weight = nn.Parameter(random_data_hidden)
        layers.append(Linear(hidden_channels, out_channels, bias, init=final_init))

        self.main = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.reshape([x.shape[0], -1, self.in_channels])
        return self.main(x)


class MultiScaleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 6, kernel_size=9, padding=4, stride=1)
        
    def forward(self, x):
        conv1_out = F.gelu(self.conv1(x))  # 改用GELU激活
        conv2_out = F.gelu(self.conv2(x))
        conv3_out = F.gelu(self.conv3(x))
        
        concatenated_output = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        return concatenated_output


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.multi_scale_conv = MultiScaleConv1D(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

        self._initialize_weights()
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.multi_scale_conv(x)
        x = self.bn1(x)
        return self.relu(x + residual)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    

class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=-1)  # 计算权重
        return (x * attn_weights).sum(dim=-1, keepdim=True)
    
