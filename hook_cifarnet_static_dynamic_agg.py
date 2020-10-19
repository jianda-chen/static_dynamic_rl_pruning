import torch.nn as nn
import torch.nn.functional as F
import torch

def return_none(*args, **kwargs):  
    return None
def identity(x):
    return x
def return_ones(x):
    x, downsampled, gates, static_gate = x
    with torch.no_grad():
        return torch.ones_like(gates), torch.ones_like(gates), 1.

class GatedConv(nn.Module):
    def __init__(self, conv_bn_relu, 
            predict_gate_with_filter_dist):
        super(GatedConv, self).__init__()
        self.conv_bn_relu = conv_bn_relu
        self.gate = nn.Linear(self.conv_bn_relu.conv.in_channels, self.conv_bn_relu.conv.out_channels)
        self.static_gate = nn.Parameter(torch.zeros(self.conv_bn_relu.conv.out_channels))
        self.predict_gate_with_filter_dist = predict_gate_with_filter_dist

        self.gate_beta = nn.Parameter(torch.zeros(self.conv_bn_relu.conv.out_channels))

        self.gate.bias.data.fill_(0.)
        self.predict_gate_with_filter_dist[-1].bias.data.fill_(1.)

        self.filter_dist = nn.Parameter(torch.zeros(self.conv_bn_relu.conv.out_channels))
        self.filter_dist.requires_grad = False
        self.filter_norm = nn.Parameter(torch.zeros(self.conv_bn_relu.conv.out_channels))
        self.filter_norm.requires_grad = False

        self.taylor_first = nn.Parameter(torch.zeros(self.conv_bn_relu.conv.out_channels))
        self.taylor_first.requires_grad = False

        self.copy_bn_weight_to_gate_bias_and_beta()

    def forward(self, x, moduel_hook1=return_none, moduel_hook2=return_none, pruning_hook1=return_ones):
        
        downsampled = F.avg_pool2d(x, x.shape[2])
        downsampled = downsampled.view(x.shape[0], x.shape[1])

        gates = self.gate(downsampled)
        filter_dist = self.filter_dist[..., None]
        filter_norm = self.filter_norm[..., None]

        static_gates = self.static_gate.clone()
        taylor_first = self.taylor_first[..., None]
        static_gates = self.predict_gate_with_filter_dist(
            torch.cat(
                (
                    static_gates[..., None], 
                    filter_dist, 
                    filter_norm, 
                    taylor_first,
                ),
                dim=1
            )
        ) # [out_channel, 1]
        static_gates = static_gates.squeeze(dim=1).unsqueeze(dim=0) # [1, out_channel]
        
        sign_gates = torch.sign(gates)
        sign_static_gates = torch.sign(static_gates)
        negative_negative = (1. - sign_gates) * (1. - sign_static_gates) / 4.
        negative_negative_to_negative = 1. - negative_negative * 2.
        negative_negative_to_negative = negative_negative_to_negative.detach()

        moduel_hook2(self.conv_bn_relu.conv)

        x = self.conv_bn_relu.conv(x)
        x = self.conv_bn_relu.bn(x)

        active, static_active, budget = pruning_hook1((x, downsampled, gates, static_gates))

        static_gates = static_gates * static_active
        gates = negative_negative_to_negative * gates * static_gates
        gates = gates * active
        gates = gates[..., None, None]
        beta = self.gate_beta[None, ...] * active * static_active

        x = x * gates + beta[..., None, None]

        x = F.relu(x)

        active_ratio = (active * static_active).mean()
        moduel_hook1(x)

        return x, active_ratio

    def copy_bn_weight_to_gate_bias_and_beta(self):
        if self.gate.bias.shape != self.conv_bn_relu.bn.weight.shape :
            raise Exception('shape mismatch: self.gate.bias.shape {} vs self.conv_bn_relu.bn.weight.shape {}'.format(
                self.gate.bias.shape,
                self.conv_bn_relu.bn.weight.shape,
            ))
        if self.gate_beta.shape != self.conv_bn_relu.bn.weight.shape :
            raise Exception('shape mismatch: self.gate_beta.shape {} vs self.conv_bn_relu.bn.weight.shape {}'.format(
                self.gate_beta.shape,
                self.conv_bn_relu.bn.weight.shape,
            ))
        # copy bn weight to gate bias
        self.gate.bias.data.copy_(self.conv_bn_relu.bn.weight.data)
        self.conv_bn_relu.bn.weight.requires_grad = False
        self.conv_bn_relu.bn.weight.data.fill_(1.)
        # copy bn bias to beta
        self.gate_beta.data.copy_(self.conv_bn_relu.bn.bias.data)
        self.conv_bn_relu.bn.bias.requires_grad = False
        self.conv_bn_relu.bn.bias.data.fill_(0.)

    def update_taylor_first(self, n_data_point:int):
        taylor_first = self.conv_bn_relu.conv.weight.data * self.conv_bn_relu.conv.weight.grad
        taylor_first = taylor_first.div_(n_data_point).detach()
        taylor_first = taylor_first.pow(2.).sum(dim=(1, 2, 3))
        taylor_first = taylor_first / (taylor_first.norm().item() + 1e-3)
        self.taylor_first.copy_(taylor_first)

    def fbs_parameters(self):
        for param in self.gate.parameters():
            yield param
        yield self.gate_beta
        for param in self.static_gate_parameters():
            yield param

    def static_gate_parameters(self):
        yield self.static_gate
        for param in self.predict_gate_with_filter_dist.parameters():
            yield param

    def gate_parameters(self):
        for param in self.gate.parameters():
            yield param

class PreserveSignMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, static_gates):
        sign = ((torch.sign(gates) + 1.) / 2.) * ((torch.sign(static_gates) + 1.) / 2.)
        sign = sign * 2 - 1.
        ctx.save_for_backward(gates, static_gates, sign)
        return sign * gates.abs() * static_gates.abs()
    
    @staticmethod
    def backward(ctx, grad_output):
        gates, static_gates, sign = ctx.saved_tensors
        return grad_output * static_gates, grad_output * gates

class MultiplyMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, active):
        result = gates * active
        ctx.save_for_backward(result, gates, active)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, gates, active = ctx.saved_tensors
        return grad_output, grad_output * gates

class GatedCifarNet(nn.Module):
    def __init__(self, cifarnet):
        super(GatedCifarNet, self).__init__()

        self.predict_gate_with_filter_dist = nn.Sequential(
            nn.Linear(4, 100),
            nn.SELU(),
            nn.Linear(100, 1)
        )
        self.cifarnet = cifarnet
        self.gconv0 = GatedConv(self.cifarnet.conv0, self.predict_gate_with_filter_dist)
        self.gconv1 = GatedConv(self.cifarnet.conv1, self.predict_gate_with_filter_dist)
        self.gconv2 = GatedConv(self.cifarnet.conv2, self.predict_gate_with_filter_dist)
        self.gconv3 = GatedConv(self.cifarnet.conv3, self.predict_gate_with_filter_dist)
        self.drop3 = self.cifarnet.drop3
        self.gconv4 = GatedConv(self.cifarnet.conv4, self.predict_gate_with_filter_dist)
        self.gconv5 = GatedConv(self.cifarnet.conv5, self.predict_gate_with_filter_dist)
        self.gconv6 = GatedConv(self.cifarnet.conv6, self.predict_gate_with_filter_dist)
        self.drop6 = self.cifarnet.drop6
        self.gconv7 = GatedConv(self.cifarnet.conv7, self.predict_gate_with_filter_dist)
        self.pool = self.cifarnet.pool
        self.fc = self.cifarnet.fc

    
    def forward(self, x, moduel_hook1=return_none, moduel_hook2=return_none, pruning_hook1=return_ones):

        xs = []
        lassos = []
        x, lasso = self.gconv0(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv1(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv2(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv3(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x = self.drop3(x)
        x, lasso = self.gconv4(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv5(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x, lasso = self.gconv6(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x = self.drop6(x)
        x, lasso = self.gconv7(x, moduel_hook1, moduel_hook2, pruning_hook1)
        lassos.append(lasso)
        xs.append(x)
        x = self.pool(x)
        x = x.view(-1, 192)

        x = self.fc(x)

        lassos = torch.stack(lassos, dim=-1)

        return  x, lassos
    def update_taylor_first(self, n_data_point):
        for c in self.children():
            if hasattr(type(c), 'update_taylor_first'):
                c.update_taylor_first(n_data_point)

    def basenet_parameters(self):
        return self.cifarnet.parameters()

    def fbs_parameters(self):
        for name, param in self.named_parameters():
            if 'gate' in name and 'gconv' in name: # and ('gate_bn' not in name):
                yield param
        # gconv0.gate.weight
        # gconv0.gate.bias
        # gconv0.gate_bn.weight
        # gconv0.gate_bn.bias
        # gconv1.gate.weight
        # gconv1.gate.bias
        # gconv1.gate_bn.weight
        # gconv1.gate_bn.bias
        # gconv2.gate.weight
        # gconv2.gate.bias
        # gconv2.gate_bn.weight
        # gconv2.gate_bn.bias
        # gconv3.gate.weight
        # gconv3.gate.bias
        # gconv3.gate_bn.weight
        # gconv3.gate_bn.bias
        # gconv4.gate.weight
        # gconv4.gate.bias
        # gconv4.gate_bn.weight
        # gconv4.gate_bn.bias
        # gconv5.gate.weight
        # gconv5.gate.bias
        # gconv5.gate_bn.weight
        # gconv5.gate_bn.bias
        # gconv6.gate.weight
        # gconv6.gate.bias
        # gconv6.gate_bn.weight
        # gconv6.gate_bn.bias
        # gconv7.gate.weight
        # gconv7.gate.bias
        # gconv7.gate_bn.weight
        # gconv7.gate_bn.bias

    @staticmethod
    def compute_loss(outputs, targets, lassos):
        loss = F.cross_entropy(outputs, targets)
        lassos_loss = 1e-8 * lassos.sum()
        loss = loss + lassos_loss
        return loss, lassos_loss


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gated=True):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, bias=False)

        # self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        return x


class CifarNet(nn.Module):
    def __init__(self, gated=True):
        super(CifarNet, self).__init__()
        self.conv0 = ConvBnRelu(3, 64, 3, padding=0,)
        self.conv1 = ConvBnRelu(64, 64, 3, padding=1,)
        self.conv2 = ConvBnRelu(64, 128, 3, padding=1, stride=2,)
        self.conv3 = ConvBnRelu(128, 128, 3, padding=1,)
        self.drop3 = nn.Dropout2d()
        self.conv4 = ConvBnRelu(128, 128, 3, padding=1,)
        self.conv5 = ConvBnRelu(128, 192, 3, padding=1, stride=2,)
        self.conv6 = ConvBnRelu(192, 192, 3, padding=1,)
        self.drop6 = nn.Dropout2d()
        self.conv7 = ConvBnRelu(192, 192, 3, padding=1,)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(192, 10)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.drop3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.drop6(x)
        x = self.conv7(x)
        x = self.pool(x)
        x = x.view(-1, 192)
        x = self.fc(x)

        return  x

c = CifarNet()
pass