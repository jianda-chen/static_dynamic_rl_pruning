import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import utils.operator

class Identity():
    pass
    
class PruningInference(nn.Module):
    def __init__(self, base_net):
        super(PruningInference, self).__init__()
        self.base_net = base_net
        self.layers_n_channels = []
        self.layer_index = None
        self.hidden = None
        self.budget_clip = False
        self.budget_clip_max = None
        self.count_kernerl = []
        self.count_output_featuremap = []
        self.count_mac = []
        self.all_channels = []
        self.dynamic_prune_ratio = None

    def sort_weights(self, module, pre_perm, post_perm=None, do_sort=True):
        assert not (do_sort is False and post_perm is not None)
        if pre_perm is not None:
            pre_perm = pre_perm.clone()
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, Identity):
            weight = module.weight.clone()
            if pre_perm is not None:
                weight = weight[:, pre_perm]
                module.weight.data.copy_(weight.data)

            weight = module.weight.clone()
            if do_sort:
                if post_perm is not None:
                    perm = post_perm
                else:
                    for_sort = weight.view(weight.shape[0], -1).pow(2).sum(dim=1)
                    _, perm = for_sort.sort(dim=0, descending=True)
                module.weight.data.copy_(weight[perm].data)
                if module.bias is not None:
                    module.bias.data.copy_(module.bias[perm].data)
                return perm
            else:
                module.weight.data.copy_(weight.data)
                return None

        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.copy_(module.weight[pre_perm].data)
            module.bias.data.copy_(module.bias[pre_perm].data)
            module.running_mean.data.copy_(module.running_mean[pre_perm].data)
            module.running_var.data.copy_(module.running_var[pre_perm].data)
            return pre_perm
        else:
            print(module)
            raise NotImplementedError()

    def count_channels(self, module):
        if isinstance(module, nn.BatchNorm2d):
            # self.layers_n_channels.append(module.num_features)
            raise NotImplementedError()
        elif isinstance(module, nn.Conv2d):
            # self.layers_n_channels.append(module.out_channels)
            self.layers_n_channels.append(module.in_channels + module.out_channels)
            self.count_kernerl.append(module.kernel_size[0] * module.kernel_size[1])
            self.all_channels.append(module.out_channels)     
        elif isinstance(module, nn.Linear):
            # self.layers_n_channels.append(module.out_features)
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def init_basenet(self, input_shape):
        s = [1, ] + list(input_shape)
        x = torch.zeros(s, device=list(self.base_net.parameters())[0].device)
        with torch.no_grad():
            self.base_net(
                    x, 
                    moduel_hook2=partial(PruningInference.count_channels, self),
                    moduel_hook1=partial(PruningInference.count_featuremap, self),
            )
            self.count_mac = [k*f for k, f in zip(self.count_kernerl, self.count_output_featuremap)]
            self.all_channels = [3,] + self.all_channels            
    def count_featuremap(self, x):
        self.count_output_featuremap.append(x.shape[2] * x.shape[3])
        

    def forward(self, x, budget, static_budget, is_train_gagent, is_train_base, with_noise=None):
        assert self.actor is not None
        batch_size = x.shape[0]
        device = x.device
        states = [] 
        actions = []
        rl_info = []
        static_info = []
        assert self.layer_index is None
        assert self.hidden is None
        if with_noise is None:
            with_noise = self.actor.training
        self.layer_index, self.hidden, noise = self.actor.zeros_state(
                batch_size, device, with_noise, is_train_gagent=is_train_gagent)
        _, self.static_hidden, noise = self.static_actor.zeros_state(
                batch_size, device, with_noise, is_train_gagent=is_train_gagent)
        # self.budget = torch.zeros(batch_size, 1, device=device).data.fill_(budget)
        self.budget = budget.clone()
        self.static_budget = static_budget.clone()
        with torch.autograd.set_grad_enabled(is_train_base):
            x = self.base_net(
                    x, 
                    pruning_hook1=partial(self.prune, 
                                states=states, 
                                actions=actions,
                                rl_info=rl_info, 
                                with_noise=with_noise,
                                is_train_gagent=is_train_gagent,
                                static_info=static_info),
            )

        self.layer_index = None
        self.hidden = None
        static_info = [
            [_[0] for _ in static_info],
            [_[1] for _ in static_info],
            [_[2] for _ in static_info],
            [_[3] for _ in static_info],
            [_[4] for _ in static_info],
            ]
        return x, states, actions, rl_info, static_info
    
    def prune(self, x, states, actions, rl_info, with_noise, is_train_gagent, static_info):
        layer_index = self.layer_index
        hidden = self.hidden
        static_hidden = self.static_hidden
        budget = self.budget
        static_budget = self.static_budget
                
        x, downsample, gates, static_gates = x
        state = torch.cat((downsample, gates), dim=-1)
        with torch.autograd.set_grad_enabled(is_train_gagent):
            action, info, budget, layer_index, hidden = self.actor.predict_action(
                        state.detach(), budget,
                        layer_index, hidden, 
                        with_noise=with_noise,
                        is_train_gagent=is_train_gagent)
            # if self.actor.training:
            rl_info.append(info)
        states.append(state.detach())
        valid_action = torch.clamp(action, min=0.1, max=1.0)
        actions.append((action, valid_action))
        if self.budget_clip:
            valid_action = torch.clamp(action, min=0.1, max=self.budget_clip_max[layer_index[0].item() - 1])
        channels = x.shape[1]
        prune_channel_begin = valid_action.detach() * channels
        mask = torch.arange(channels, dtype=torch.float32).to(x.device)
        mask = mask.repeat(x.shape[0], 1)
        mask = utils.operator.convert_n10p1_to_01(torch.sign(prune_channel_begin - mask), zero_to_p1=True)
        mask[:, 0] = 1. # makesure at least one channel is kept 
        sorted_index = torch.argsort(gates, dim=1, descending=True)
        masked_sorted_index = sorted_index * mask.long() + sorted_index[:, 0:1] * (1 - mask.long())
        active = torch.zeros_like(gates)
        active.scatter_(1, masked_sorted_index, 1.)  

        ##### for static pruning
        with torch.autograd.set_grad_enabled(is_train_gagent):
            static_action, info, static_budget, layer_index, static_hidden = self.static_actor.predict_action(
                        static_budget,
                        self.layer_index, static_hidden, 
                        with_noise=with_noise,
                        is_train_gagent=is_train_gagent)
        static_valid_action = torch.clamp(static_action, min=0.1, max=1.0)
        channels = x.shape[1]
        prune_channel_begin = static_valid_action.detach() * channels
        mask = torch.arange(channels, dtype=torch.float32).to(x.device)
        mask = mask.repeat(x.shape[0], 1)
        # mask = (torch.sign(prune_channel_begin - mask) + 1. )  / 2.
        mask = utils.operator.convert_n10p1_to_01(torch.sign(prune_channel_begin - mask), zero_to_p1=True)
        mask[:, 0] = 1. # makesure at least one channel is kept 
        static_gates = static_gates
        static_sorted_index = torch.argsort(static_gates, dim=1, descending=True)
        # static_sorted_index = torch.argsort(gates, dim=1, descending=True)
        static_masked_sorted_index = static_sorted_index * mask.long() + static_sorted_index[:, 0:1] * (1 - mask.long())

        static_active = torch.zeros_like(static_gates)
        static_active = static_active.repeat((x.shape[0], 1))
        static_active.scatter_(1, static_masked_sorted_index, 1.)  
        
        active, static_active, overlap_inactive = self.resolve_overlap_mask(
                active, static_active, gates, static_gates,
                valid_action, static_valid_action, channels, 
                dynamic_prune_ratio=self.dynamic_prune_ratio)
        # print(active[0].mean().item(), static_active[0].mean().item())

        static_info.append(((static_action, static_valid_action), info,
                (active*static_active).mean(dim=-1),
                overlap_inactive.float().mean(dim=-1),
                ((1. - active) * (1. - static_active)).float().mean(dim=-1),
                ))


        self.layer_index = layer_index
        self.hidden = hidden
        self.budget = budget.detach()# - 1. / self.actor.layer_index_size * valid_action.data
        return active.detach(), static_active.detach(), budget[0][0].item()

    def resolve_overlap_mask(self, active, static_active, gates, static_gates,
                valid_action, static_valid_action, channels, dynamic_prune_ratio):
        batch_size = gates.shape[0]
        overlap_inactive = (1. - active) * (1. - static_active)
        overlap_inactive_ratio = overlap_inactive.mean(dim=1, keepdim=True)
        valid_action = 1. - (1. - valid_action - overlap_inactive_ratio) * dynamic_prune_ratio - overlap_inactive_ratio
        static_valid_action = 1. - (1. - static_valid_action - overlap_inactive_ratio) * (1. - dynamic_prune_ratio) - overlap_inactive_ratio
        valid_action = torch.clamp(valid_action, 0.,  1.,)
        static_valid_action = torch.clamp(static_valid_action, 0.,  1.,)
        gates = gates.clone().detach()
        static_gates = static_gates.clone().detach()
        static_gates = static_gates.repeat(batch_size, 1)
        if hasattr(overlap_inactive, 'bool'):
            overlap_inactive_bool = overlap_inactive.bool()
        else:
            overlap_inactive_bool = overlap_inactive.byte()
        gates.masked_fill_(overlap_inactive_bool, torch.finfo(gates.dtype).min)
        static_gates.masked_fill_(overlap_inactive_bool, torch.finfo(static_gates.dtype).min)

        #### for dynamic prunning
        prune_channel_begin = valid_action.detach() * channels
        mask = torch.arange(channels, dtype=torch.float32).to(valid_action.device)
        mask = mask.repeat(batch_size, 1)
        mask = utils.operator.convert_n10p1_to_01(torch.sign(prune_channel_begin - mask), zero_to_p1=True)
        mask[:, 0] = 1. # makesure at least one channel is kept 
        sorted_index = torch.argsort(gates, dim=1, descending=True)
        masked_sorted_index = sorted_index * mask.long() + sorted_index[:, 0:1] * (1 - mask.long())

        active = torch.zeros_like(gates)
        active.scatter_(1, masked_sorted_index, 1.) 

        ##### for static pruning
        prune_channel_begin = static_valid_action.detach() * channels
        mask = torch.arange(channels, dtype=torch.float32).to(static_valid_action.device)
        mask = mask.repeat(batch_size, 1)
        mask = utils.operator.convert_n10p1_to_01(torch.sign(prune_channel_begin - mask), zero_to_p1=True)
        mask[:, 0] = 1. # makesure at least one channel is kept 
        static_sorted_index = torch.argsort(static_gates, dim=1, descending=True)
        static_masked_sorted_index = static_sorted_index * mask.long() + static_sorted_index[:, 0:1] * (1 - mask.long())

        static_active = torch.zeros_like(static_gates)
        static_active.scatter_(1, static_masked_sorted_index, 1.)  

        return active, static_active, overlap_inactive

    
