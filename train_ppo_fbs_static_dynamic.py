import torch

# import multiprocessing
# multiprocessing.set_start_method('spawn',True)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import os
import time
import argparse
import hook_cifarnet_static_dynamic_agg as hook_cifarnet
from BudgetPruning_static_dynamic import PruningInference
from PPOAgentGaussian import GaussianActorCritic
from PPOAgentGaussian_static import GaussianActorCritic_static
from utils.util_tee import Tee

import tqdm
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--budget', type=float, default=0.5)
parser.add_argument('--dpratio', type=float)
program_args = parser.parse_args()

arg = {}
arg['sigma'] = 0.02


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_train = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=True, download=True, transform=transform_train)
data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

data_test = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=False, download=True, transform=transform_test)
data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, num_workers=2)

# model_name = 'cifarnet'
# model_name = 'resnet56'

with torch.cuda.device(0):
    device = torch.cuda.current_device()

    net = hook_cifarnet.GatedCifarNet(hook_cifarnet.CifarNet())
    forked_basenet = hook_cifarnet.GatedCifarNet(hook_cifarnet.CifarNet())
    print(net.load_state_dict(torch.load('../save_models/static_dynamic_agg/static_dynamic_agg.p')['net']))
    

    # net = net.module
    net.cuda(torch.cuda.current_device())
    forked_basenet.cuda(torch.cuda.current_device())
    criterion = nn.CrossEntropyLoss()

    pruning_net = PruningInference(base_net=net)
    pruning_net.to(device)
    pruning_net.init_basenet(input_shape=(3, 32, 32))
    gaussian_actor = GaussianActorCritic(rnn_hidden_size=128,
                                    prunable_layers_n_channels=pruning_net.layers_n_channels,
                                    sigma=arg['sigma'],)
    static_actor = GaussianActorCritic_static(rnn_hidden_size=128,
                                    prunable_layers_n_channels=pruning_net.layers_n_channels,
                                    sigma=arg['sigma'],)

    gaussian_actor.to(device)
    static_actor.to(device)
    pruning_net.actor = gaussian_actor
    pruning_net.static_actor = static_actor
    

    optimizer = optim.Adam(net.parameters(), lr=1e-3)

arg['budget'] = program_args.budget
arg['static_budget'] = arg['budget']
arg['dynamic_prune_ratio'] = program_args.dpratio


model_name = 'cifarnet_budget_{}_dpratio_{}'.format(arg['budget'], arg['dynamic_prune_ratio'])
save_dir = '../saved_ppo_static_dynamic_rnn/{}'.format(model_name)
os.makedirs(save_dir, exist_ok=True)
log_dir = os.path.join(save_dir, 'log')
if os.path.exists(log_dir):
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('{} exists'.format(log_dir))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
writer = SummaryWriter(log_dir=log_dir)
tee = Tee(os.path.join(save_dir, 'tee.txt'), 'a')


arg['total_mac'] = 0. 
print(pruning_net.count_mac)
for i in range(len(pruning_net.count_mac)):
    arg['total_mac'] += pruning_net.count_mac[i] * pruning_net.all_channels[i] * pruning_net.all_channels[i+1]

arg['target_mac'] = 0.
all_channels = [3., ] + [c * arg['budget'] for c in pruning_net.all_channels[1:]]
for i in range(len(pruning_net.count_mac)):
    arg['target_mac'] += pruning_net.count_mac[i] * all_channels[i] * all_channels[i+1]
arg['comsuption_cap'] = arg['target_mac'] / arg['total_mac']

print("arg['total_mac']", arg['total_mac'])
print("arg['target_mac']", arg['target_mac'])
print("arg['comsuption_cap']", arg['comsuption_cap'])


arg['total_param'] = 0. 
for i in range(len(pruning_net.count_kernerl)):
    arg['total_param'] += pruning_net.count_kernerl[i] * pruning_net.all_channels[i] * pruning_net.all_channels[i+1]

arg['target_param'] = 0.
all_channels = [3., ] + [c * arg['budget'] for c in pruning_net.all_channels[1:]]
for i in range(len(pruning_net.count_kernerl)):
    arg['target_param'] += pruning_net.count_kernerl[i] * all_channels[i] * all_channels[i+1]
arg['param_cap'] = arg['target_param'] / arg['total_param']
print("arg['total_param']", arg['total_param'])
print("arg['target_param']", arg['target_param'])
print("arg['param_cap']", arg['param_cap'])

pruning_net.dynamic_prune_ratio = arg['dynamic_prune_ratio'] 


arg['p'] = 1. #- 0.1 / gaussian_actor.layer_index_size
arg['alpha'] = 1.
arg['gamma'] = 0.999
arg['gae_lambda'] = 0.95

running_mean_loss00 = {'mean': None, 'mean_step':0, 'mean_static': None, 'mean_static_step':0}

def train_rl(images, labels, sample_budget, static_budget, actor_optimizer, static_actor_optimizer,
                loss_list, batch_list, r_list, a_list, total, correct):
    net.eval()
    gaussian_actor.train()
    static_actor.train()
    # rewards
    with torch.no_grad():
        output0, states_list, actions_list, rl_info_list, static_info = \
        pruning_net(images, sample_budget, static_budget, is_train_gagent=False, is_train_base=False)
        output0, active_ratio = output0
        loss00 = F.cross_entropy(output0, labels, reduction='none')
        _, predicted = torch.max(output0.data, -1)
        loss0 = torch.exp(-loss00)
        crt = (predicted == labels).float()
        valid_actions = [a[1] for a in actions_list]
        actions_list = [a[0] for a in actions_list]
        LEN_eps_minus_one = len(valid_actions)

        performance_reward_list = []
        valid_actions = [a.detach() for a in valid_actions]
        budget_reward_list = []
        for i in range(LEN_eps_minus_one):
            budget_reward_list.append(torch.zeros_like(valid_actions[i]))

        ratio_detach = [torch.ones_like(valid_actions[0]),] + valid_actions
        ratio_detach = torch.stack(ratio_detach, dim=0)
        ratio_detach = torch.clamp(ratio_detach, min=0.0, max=1.0)
        count_mac = torch.tensor(pruning_net.count_mac, device=ratio_detach.device, dtype=torch.float)
        all_channels = torch.tensor(pruning_net.all_channels, device=ratio_detach.device, dtype=torch.float)
        count_mac = count_mac[..., None, None]
        all_channels = all_channels[..., None, None]
        mac_consumption = (count_mac * torch.round(all_channels[:-1] * ratio_detach[:-1])
                    * torch.round(all_channels[1:] * ratio_detach[:-1])).sum(dim=0)
        budgets_consumption = mac_consumption / arg['total_mac']

        out_of_budget_soft = torch.sign(-arg['comsuption_cap'] + budgets_consumption)
        # translate {-1, 1} to {0, 1}
        out_of_budget_soft = (out_of_budget_soft + 1.) * 0.5
        budget_reward_list[-1] = budget_reward_list[-1] + \
                out_of_budget_soft * (-1.2) * (torch.exp((budgets_consumption - arg['comsuption_cap']) / 0.03) - 1.) + \
                (1. - out_of_budget_soft) * (0.1) * (arg['comsuption_cap'] - budgets_consumption)

        for i in range(LEN_eps_minus_one):
            performance_reward = torch.zeros_like(budget_reward_list[i])
            if i == LEN_eps_minus_one - 1:
                if running_mean_loss00['mean_step'] == 0:
                    running_mean_loss00['mean']  = loss00.mean().item()
                    running_mean_loss00['mean_step'] += 1
                elif running_mean_loss00['mean_step'] <= 200:
                    running_mean_loss00['mean'] = (running_mean_loss00['mean']  * (1.-1./ running_mean_loss00['mean_step']) 
                        + loss00.mean().item() *  1. / running_mean_loss00['mean_step'])
                    running_mean_loss00['mean_step'] += 1
                else:
                    running_mean_loss00['mean']  = running_mean_loss00['mean']  * 0.995 + loss00.mean().item() * 0.005
                performance_reward += - (loss00[..., None] / running_mean_loss00['mean'] )
            performance_reward_list.append(performance_reward)
        performance_reward_list = [r.detach() for r in performance_reward_list]
        rewards = [p.data + arg['p'] * b.data 
                for p, b in zip(performance_reward_list, budget_reward_list)]

        # GAE
        values = [info[1] for info in rl_info_list]
        values.append(torch.zeros_like(values[-1]))
        R = torch.zeros_like(values[-1])
        Rs = []
        gae = torch.zeros_like(values[-1])
        gaes = []
        for j in reversed(range(LEN_eps_minus_one)):
            R = arg['gamma'] * R + rewards[j]
            delta_t = rewards[j] + arg['gamma'] * \
                    values[j + 1] - values[j]
            gae = gae * arg['gamma'] * arg['gae_lambda'] + delta_t
            Rs.append(R)
            gaes.append(gae)
        Rs = list(reversed(Rs))
        gaes = list(reversed(gaes))

        # convert to tensor
        Rs = torch.stack(Rs)
        gaes = torch.stack(gaes)
        old_values = torch.stack(values[:-1])
        old_action = torch.stack(actions_list)
        old_a_dist_mean = torch.stack([info[0].mean for info in rl_info_list])
        old_a_dist_stddev = torch.stack([info[0].stddev for info in rl_info_list])
        

    # PPO config 
    action_range_min = 0.1
    action_range_max = 1.0
    vf_coef = 0.5
    lam = 1.
    ent_coef = 0.0
    cliprange = 0.2
    out_range_coef = 100.
    nbatch = images.shape[0]
    nbatch_train = 64
    assert nbatch % nbatch_train == 0
    noptepochs = 1 #2 #4
    # PPO 
    for _ in range(noptepochs):
        for start in range(0, nbatch, nbatch_train):
            end = start + nbatch_train
            ppo_states = [s[start:end] for s in states_list]
            ppo_sample_budget = sample_budget[start:end]
            ppo_a_dist_mean = []
            ppo_a_dist_stddev = []
            ppo_values = []
            # forward agent 
            layer_index, hidden, _ = gaussian_actor.zeros_state(
                    nbatch_train, 
                    ppo_states[0].device, 
                    with_noise=False, 
                    is_train_gagent=None)
            for i in range(LEN_eps_minus_one):
                _, (ppo_a_dist, ppo_v), _, layer_index, hidden = gaussian_actor.predict_action(
                    state=ppo_states[i],
                    budget=ppo_sample_budget,
                    layer_index=layer_index,
                    hidden=hidden,
                    with_noise=True,
                    is_train_gagent=None
                )
                ppo_a_dist_mean.append(ppo_a_dist.mean)
                ppo_a_dist_stddev.append(ppo_a_dist.stddev)                
                ppo_values.append(ppo_v)
            # loss 
            ppo_a_dist_mean = torch.stack(ppo_a_dist_mean)
            ppo_a_dist_stddev = torch.stack(ppo_a_dist_stddev)
            ppo_action_dist = torch.distributions.normal.Normal(
                loc=ppo_a_dist_mean, 
                scale=ppo_a_dist_stddev, 
            )
            ppo_vpred  = torch.stack(ppo_values)
            ppo_ADV = gaes[:, start:end]
            ppo_ADV = (ppo_ADV - ppo_ADV.mean()) / ppo_ADV.std()
            ppo_R = Rs[:, start:end]
            ppo_OLDVPRED = old_values[:, start:end]
            ppo_action = old_action[:, start:end]
            ppo_old_action_dist = torch.distributions.normal.Normal(
                loc=old_a_dist_mean[:, start:end], 
                scale=old_a_dist_stddev[:, start:end], 
            )
            ppo_OLDLOGPAC = ppo_old_action_dist.log_prob(ppo_action)
            # vf
            vf_losses1 = torch.pow(ppo_vpred - ppo_R, 2)
            vpredclipped = ppo_OLDVPRED + torch.clamp(ppo_vpred - ppo_OLDVPRED, -cliprange, cliprange)
            vf_losses2 = torch.pow(vpredclipped - ppo_R, 2)
            vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
            # pg
            # print(ppo_action.shape, ppo_action_dist.mean.shape)
            ppo_logpac = ppo_action_dist.log_prob(ppo_action)
            ratio = torch.exp(ppo_logpac - ppo_OLDLOGPAC)
            pg_losses = -ppo_ADV * ratio
            pg_losses2 = -ppo_ADV * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
            pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))

            entropy = torch.mean(ppo_action_dist.entropy())
            # action out of range penalty
            out_of_range = F.smooth_l1_loss(ppo_a_dist_mean, 
                        torch.clamp(ppo_a_dist_mean.detach(), action_range_min, action_range_max))
            # Total loss
            actor_loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + out_range_coef * out_of_range
            # print(pg_loss, entropy, vf_loss)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(gaussian_actor.parameters(), .5) #.5
            actor_optimizer.step()
    ###### static prunning
    static_actor_loss, static_rewards = GaussianActorCritic_static.PPO_loss(
        static_actor, static_info[0], static_info[1], loss00, crt, static_budget,
        static_actor_optimizer, arg, pruning_net,
        batchsize=images.shape[0], running_mean_loss00=running_mean_loss00)

    static_valid_actions = [a[1] for a in static_info[0]]
    static_valid_actions = torch.stack(static_valid_actions)
    static_valid_actions = torch.mean(static_valid_actions[:, 0])
    static_rewards = torch.mean(torch.sum(torch.stack(static_rewards), dim=0), dim=0)

    # acc
    _, predicted = output0.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
    
    active_ratio = static_info[2]
    active_ratio = torch.stack(active_ratio)
    active_ratio = torch.mean(active_ratio[:, 0])
    loss0 = loss0.detach().mean()
    loss_list.append(loss0.detach().cpu().item())
    batch_list.append(i+1)

    with torch.no_grad():
        action = torch.stack(valid_actions)
        action = torch.mean(action[:, 0])
        rewards = torch.mean(torch.sum(torch.stack(rewards), dim=0), dim=0)
    
    r_list.append(rewards.detach().cpu().item())
    a_list.append(action.detach().cpu().item())

    return action, static_valid_actions, loss0, active_ratio, rewards, static_rewards, total, correct, actions_list

def train(epoch, actor_optimizer, static_actor_optimizer, optimizer, train_base):
# def train_gagent(epoch):
    global cur_batch_win
    loss_list, batch_list, r_list, a_list = [], [], [], []
    total = 0
    correct = 0
    pbar = tqdm.tqdm(data_train_loader, ascii=True, ncols=115, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")
    # for param in net.parameters():
    #     param.requires_grad = False
    for i, (images, labels) in enumerate(pbar):
        images = images.cuda()
        labels = labels.cuda()
        loss = 0.0

        sample_budget = torch.ones([images.shape[0], 1]).to(images.device) * arg['budget']
        static_budget = torch.ones([images.shape[0], 1]).to(images.device) * arg['static_budget']
        # train rl
        action, static_valid_actions, loss0, active_ratio, rewards, static_rewards, total, correct, actions_list = \
            train_rl(images, labels, sample_budget, static_budget, actor_optimizer, static_actor_optimizer,
                loss_list, batch_list, r_list, a_list, total, correct)
        be = pruning_net.budget[0].cpu().item()
        # train basenet
        # if train_base and i % 10 == 0:
        if train_base and i % 2 == 0:
            loss = train_basenet_one_batch(images, labels, sample_budget, static_budget, optimizer)
        pbar.set_description('Train-Ep {}'.format(epoch))
        pbar.set_postfix_str('b={:.3f} sb={:.3f}|a={:.3f} sa={:.3f} l={:.3f} r={:.3f} sr={:.3f}|ar={:.3f}'.format(
                        sample_budget[0].cpu().item(),
                        static_budget[0].cpu().item(),
                        # be,
                        action.detach().cpu().item(),
                        static_valid_actions.detach().cpu().item(),
                        loss0.detach().cpu().item(),
                        rewards.detach().cpu().item(),
                        static_rewards.detach().cpu().item(),
                        active_ratio.detach().cpu().item(),
                        ))
    loss_list = torch.tensor(loss_list)
    r_list = torch.tensor(r_list)
    a_list = torch.tensor(a_list)
    print('Train - Epoch {} complete. loss: {:.6g} {:.6g}, r:{:.6g} {:.6g}, a:{:.6g} {:.6g}, acc:Acc: {:.3f} % ({:d}/{:d})'.format(
                epoch, 
                loss_list.mean(), loss_list.std(), 
                r_list.mean(), r_list.std(), 
                a_list.mean(), a_list.std(), 
                100.*correct/total, correct, total,
                ))

def train_basenet_one_batch(images, labels, sample_budget, static_budget, optimizer):
    pruning_net.base_net.train()
    pruning_net.actor.train()

    output, _, _, _, _ = pruning_net(images, sample_budget, static_budget, is_train_gagent=False, is_train_base=True)
    output, _ = output
    loss = criterion(output, labels)

    # loss_list.append(loss.detach().cpu().item())
    # batch_list.append(i+1)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pruning_net.base_net.parameters(), 40.)
    optimizer.step()

    return loss.cpu().item()

def train_basenet(epoch, optimizer):
    global cur_batch_win
    net.train()
    gaussian_actor.eval()
    loss_list, batch_list, r_list, a_list = [], [], [], []
    pbar = tqdm.tqdm(data_train_loader, ascii=True, ncols=115, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")
    # for param in net.parameters():
    #     param.requires_grad = False
    for i, (images, labels) in enumerate(pbar):
        images = images.cuda()
        labels = labels.cuda()
        loss = 0.0
        
        sample_budget = torch.ones([images.shape[0], 1]).to(images.device) * arg['budget']
        static_budget = torch.ones([images.shape[0], 1]).to(images.device) * arg['static_budget']
        
        output, _, actions_list, _, static_info = pruning_net(images, sample_budget, static_budget, is_train_gagent=False, is_train_base=True)
        output, _ = output

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        with torch.no_grad():
            actions_list = [a[0] for a in actions_list]

            action = torch.stack(actions_list)
            action = torch.mean(action[:, 0])
            static_valid_actions = [a[1] for a in static_info[0]]
            static_valid_actions = torch.stack(static_valid_actions)
            static_valid_actions = torch.mean(static_valid_actions[:, 0])


        pbar.set_description('Train B Ep {}'.format(epoch))
        # pbar.set_postfix(Loss=loss.detach().cpu().item(), iteration=i)
        pbar.set_postfix_str('b={:.3f} sb={:.3f}|a={:.3f} sa={:.3f} l={:.3f} r={:.3f} | l={:.3f}'.format(
                sample_budget[0].cpu().item(),
                static_budget[0].cpu().item(),
                # be,
                action.detach().cpu().item(),
                static_valid_actions.detach().cpu().item(),
                i,
                0,
                0,
                loss
                ))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 40.)
        optimizer.step()
    print('Train BaseNet - Epoch {} complete. loss: {:.6g}'.format(
                epoch, 
                torch.tensor(loss_list).mean(),
                ))

def test(data_test_loader, sample_budget, print_stat=False):
    pruning_net.base_net.eval()
    pruning_net.actor.eval()
    total_correct = 0
    avg_loss = 0.0
    avg_action = 0.0
    sparsity = []
    action_dist_list = []
    static_sparsity = []
    static_action_dist_list = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.cuda()
            labels = labels.cuda()
            if sample_budget is None:
                sample_budget_ones = torch.distributions.Uniform(arg['budget'], high=1.
                        ).sample(torch.Size([images.shape[0], 1])).to(images.device)
            else:
                sample_budget_ones = torch.ones(images.shape[0], 1, device=images.device) * sample_budget
            static_budget = torch.ones(images.shape[0], 1, device=images.device) * arg['static_budget']
            output, states, actions_list, action_p_list, static_info = pruning_net(images, sample_budget_ones, static_budget,
                    is_train_gagent=False, is_train_base=False, with_noise=False)
            output, _ = output
            actions_list = [a[1] for a in actions_list]
            action = torch.stack(actions_list)
            action = torch.mean(action)
            avg_action += action * images.shape[0]
            # avg_loss += criterion(output, labels).sum() * images.shape[0]
            avg_loss += criterion(output, labels).item()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            sparsity.append([a.detach().cpu() for a in actions_list])
            action_dist = [ap[0] for ap in action_p_list]
            action_dist_list.append(action_dist)
            # print(len(action_dist))
            ##### for static pruning
            static_sparsity.append([a[1].detach().cpu() for a in static_info[0]])
            static_action_dist = [ap[0] for ap in static_info[1]]
            static_action_dist_list.append(static_action_dist)

    # avg_loss /= len(data_test_loader.dataset)
    avg_loss /= len(data_test_loader)
    avg_action /= len(data_test_loader.dataset)
    print('Test Avg. Loss: {}, Accuracy: {}, action: {}, budget: {}'.format(
            avg_loss, 
            float(total_correct) / len(data_test_loader.dataset),
            avg_action.detach().cpu().item(),
            sample_budget if sample_budget is not None else '[{}, {}]'.format(arg['budget'], 1.),
            ))
    if print_stat:
        writer.add_scalar('training_test/loss', avg_loss)
        writer.add_scalar('training_test/Accuracy', float(total_correct) / len(data_test_loader.dataset))
        writer.add_scalar('training_test/action', avg_action.detach().cpu().item())
        writer.add_scalar('training_test/static_action', avg_action.detach().cpu().item())
        writer.add_scalar('training_test/budget', arg['budget'])
    else:
        writer.add_scalar('testing_test/loss', avg_loss)
        writer.add_scalar('testing_test/Accuracy', float(total_correct) / len(data_test_loader.dataset))
        writer.add_scalar('testing_test/action', avg_action.detach().cpu().item())
        writer.add_scalar('traing_testing_testtest/static_action', avg_action.detach().cpu().item())
        writer.add_scalar('testing_test/budget', arg['budget'])

    sparsity = [torch.cat([s[i] for s in sparsity], dim=0) for i in range(len(sparsity[0]))]
    action_dist_mu = [torch.cat([d[i].mean for d in action_dist_list], dim=0) for i in range(len(action_dist_list[0]))]
    mean_sparsity = []
    std_sparsity = []
    std_mu = []
    mean_mu = []
    for i in range(gaussian_actor.layer_index_size):
        # mean_s = torch.mean(torch.stack([s[i] for s in sparsity], dim=0))
        mean_s = torch.mean(sparsity[i])
        mean_sparsity.append(mean_s.data.item())
        std_s = torch.std(sparsity[i])
        std_sparsity.append(std_s.data.item())
        std_mu.append(torch.std(action_dist_mu[i]).data.item())
        mean_mu.append(torch.mean(action_dist_mu[i]).data.item())
    
    if print_stat:
        print('Test Avg. Sparsity ' , end='')
        for i, s in enumerate(mean_sparsity):
            print('{}: {}, '.format(i, s), end='')
            writer.add_scalar('training_test/{}_dynamic_parsity'.format(i), s)
        print('')
        print('Test Avg. Sparsity STD ' , end='')
        for i, s in enumerate(std_sparsity):
            print('{}: {}, '.format(i, s), end='')
        print('')
        print('Test Avg. action mu mean ' , end='')
        for i, s in enumerate(mean_mu):
            print('{}: {}, '.format(i, s), end='')
            writer.add_scalar('training_test/{}_dynamic_action'.format(i), s)
        print('')
        print('Test Avg. action mu STD ' , end='')
        for i, s in enumerate(std_mu):
            print('{}: {}, '.format(i, s), end='')
        print('\n')



    ##### for static pruning

    sparsity = [torch.cat([s[i] for s in static_sparsity], dim=0) for i in range(len(static_sparsity[0]))]
    action_dist_mu = [torch.cat([d[i].mean for d in static_action_dist_list], dim=0) for i in range(len(static_action_dist_list[0]))]
    mean_sparsity = []
    std_sparsity = []
    std_mu = []
    mean_mu = []
    for i in range(gaussian_actor.layer_index_size):
        # mean_s = torch.mean(torch.stack([s[i] for s in sparsity], dim=0))
        mean_s = torch.mean(sparsity[i])
        mean_sparsity.append(mean_s.data.item())
        std_s = torch.std(sparsity[i])
        std_sparsity.append(std_s.data.item())
        std_mu.append(torch.std(action_dist_mu[i]).data.item())
        mean_mu.append(torch.mean(action_dist_mu[i]).data.item())
    
    if print_stat:
        print('Test Avg. Static Sparsity ' , end='')
        for i, s in enumerate(mean_sparsity):
            print('{}: {}, '.format(i, s), end='')
            writer.add_scalar('training_test/{}_static_parsity'.format(i), s)
        print('')
        print('Test Avg. Static Sparsity STD ' , end='')
        for i, s in enumerate(std_sparsity):
            print('{}: {}, '.format(i, s), end='')
        print('')
        print('Test Avg. Static action mu mean ' , end='')
        for i, s in enumerate(mean_mu):
            print('{}: {}, '.format(i, s), end='')
            writer.add_scalar('training_test/{}_static_action'.format(i), s)
        print('')
        print('Test Avg. Static action mu STD ' , end='')
        for i, s in enumerate(std_mu):
            print('{}: {}, '.format(i, s), end='')
        print('\n')


    return float(total_correct) / len(data_test_loader.dataset), mean_sparsity

def train_and_test(epoch, rl_ite, base_ite,):
    print("arg['budget']: {}".format(arg['budget']))
    print("arg['dynamic_prune_ratio']: {}".format(arg['dynamic_prune_ratio']))

    for i in range(rl_ite):
        e_str = '{}-{}-{}'.format(epoch, 0, i)
        writer.add_scalars('phase', {'phase1': epoch, 'phase2': 0, 'epoch':i})
        train(e_str, actor_optimizer, static_actor_optimizer, optimizer, train_base=False)
        acc, _ = test(data_test_loader, sample_budget=arg['budget'])
    acc, sparsity = test(data_train_loader, sample_budget=arg['budget'], print_stat=True)
    
    for i in range(base_ite):
        e_str = '{}-{}-{}'.format(epoch, 1, i)
        writer.add_scalars('phase', {'phase1': epoch, 'phase2': 1, 'epoch':i})
        train_basenet(e_str, optimizer)
        acc, _ = test(data_test_loader, sample_budget=arg['budget'])
    acc, sparsity = test(data_train_loader, sample_budget=arg['budget'], print_stat=True)
    
    save_dir = '../saved_ppo_static_dynamic/{}'.format(model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 
        '{}_epoch_{}_budget_{}.pt'.format(model_name, epoch, arg['budget']))
    torch.save(
        {
            'pruning_net': pruning_net.state_dict(),
            'actor_optimizer': actor_optimizer.state_dict(),
            'mean_sparsity': sparsity,
            'basenet_optimizer': optimizer.state_dict(),
            'epoch': epoch,
        },
        save_path
    )

    
class FillZeroList():
    def __init__(self, head, fill=-1):
        self.head = head
        self.len = len(self.head)
        self.fill = fill
    
    def __getitem__(self, idx):
        if idx >= self.len:
            return self.fill
        else:
            return self.head[idx]
    

def main():


    rl_ite = FillZeroList([ 4, ], fill=4)
    base_ite = FillZeroList([2,] * 10+ [200,], fill=2)
    for e in range(0, 32):
        train_and_test(e, rl_ite[e], base_ite[e]
    acc = test(data_test_loader)

if __name__ == '__main__':
    main()

