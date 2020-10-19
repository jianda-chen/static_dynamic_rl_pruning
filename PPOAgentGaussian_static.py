import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianActorCritic_static(nn.Module):
    def __init__(self, rnn_hidden_size, prunable_layers_n_channels, sigma=0.05):
        super(GaussianActorCritic_static, self).__init__()
        self.prunable_layers_n_channels = prunable_layers_n_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.sigma = sigma

        self.layer_index_size = len(prunable_layers_n_channels)
        self.max_prunable_layers_n_channels = max(self.prunable_layers_n_channels)
        self.encoder_size = self.rnn_hidden_size # could be different
        # self.rnn_input_size = self.encoder_size + self.layer_index_size
        self.rnn_input_size = self.encoder_size + 1 # 1 for budget

        self.static_rnn = nn.LSTM(self.rnn_hidden_size, self.rnn_hidden_size, num_layers=1)

        self.static_encoder = nn.Sequential(
                    nn.Linear(1, self.rnn_hidden_size),
                    nn.ReLU(),
                )

        self.static_decoder_mu = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, 1),
        )
        self.fc = nn.Linear(self.rnn_hidden_size, self.rnn_hidden_size)
        self.static_decoder_log_sigma = nn.Linear(self.rnn_hidden_size, 1)
        self.static_decoder_mu[0].bias.data.fill_(0.)
        self.static_decoder_mu[0].weight.data *= 0.2
        self.static_decoder_log_sigma.bias.data.fill_(-4.)
        self.static_decoder_value = nn.Linear(self.rnn_hidden_size , 1)

        self.running_mean_loss00 = {'mean': None}

    def create_encoder(self, in_channels):
        return nn.Sequential(
                    nn.Linear(in_channels, self.encoder_size),
                    nn.ReLU(),
                )
    
    def reset_actor(self, batch_size=None, hidden=None):

        if hidden is not None:
            self.hidden = hidden
        else:
            self.hidden = (torch.zeros(self.rnn.num_layers, 
                                    batch_size, self.rnn_hidden_size).to(device),
                           torch.zeros(self.rnn.num_layers,    
                                    batch_size, self.rnn_hidden_size).to(device)
                            )
        self.layer_index = torch.zeros(batch_size, 1, dtype=torch.int32)
        self.batch_size = batch_size

        self.state_list = []

    def zeros_state(self, batch_size, device, with_noise, is_train_gagent=None):
        with torch.cuda.device(device):
            static_hidden = (torch.zeros(self.static_rnn.num_layers, 
                            batch_size, self.rnn_hidden_size, device=device),
                    torch.zeros(self.static_rnn.num_layers,    
                            batch_size, self.rnn_hidden_size, device=device)
                    )
            layer_index = 0
        return layer_index, static_hidden, None

    def predict_action(self, budget, layer_index, static_hidden, with_noise=True, is_train_gagent=None):
        
        x = torch.ones((static_hidden[0].shape[1], 1), device=static_hidden[0].device) \
                * self.prunable_layers_n_channels[layer_index] \
                / self.max_prunable_layers_n_channels
        x = self.static_encoder(x)

        x, static_hidden = self.static_rnn(x.unsqueeze(0), 
                                static_hidden)
        x = x.squeeze(0)
        static_action_mu = self.static_decoder_mu(x)
        static_action_mu = static_action_mu + budget
        static_action_log_sigma = self.static_decoder_log_sigma(x)
        static_value = self.static_decoder_value(x)
        gaussian_dist = torch.distributions.normal.Normal(static_action_mu, 
                scale=torch.exp(static_action_log_sigma)
                )
        if with_noise:
            action = gaussian_dist.rsample().detach()
        else:
            action = static_action_mu.detach()

        with torch.no_grad():
            layer_index = layer_index + 1

        return action, (gaussian_dist, static_value), budget, layer_index, static_hidden

    def forward():
        raise NotImplementedError

    @staticmethod
    def PPO_loss(static_actor, actions_list, rl_info_list, loss00, crt, sample_budget, 
            actor_optimizer, arg, pruning_net, batchsize, running_mean_loss00):
        action_range_min = 0.1
        action_range_max = 1.0
        with torch.no_grad():
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
            all_channels = torch.tensor(pruning_net.all_channels, device=ratio_detach.device, dtype=ratio_detach.dtype)
            all_channels = all_channels[..., None, None]
            count_kernerl = torch.tensor(pruning_net.count_kernerl, device=ratio_detach.device, dtype=ratio_detach.dtype)
            count_kernerl = count_kernerl[..., None, None]
            param_consumption = (count_kernerl * torch.round(all_channels[:-1] * ratio_detach[:-1])
                    * torch.round(all_channels[1:] * ratio_detach[:-1])).sum(dim=0)
            budgets_consumption = param_consumption / arg['total_param']

            out_of_budget_soft = torch.sign(-arg['param_cap'] + budgets_consumption)
            # translate {-1, 1} to {0, 1}
            out_of_budget_soft = (out_of_budget_soft + 1.) * 0.5
            budget_reward_list[-1] = budget_reward_list[-1] + \
                    out_of_budget_soft * (-1.2) * (torch.exp((budgets_consumption - arg['param_cap']) / 0.03) - 1.) + \
                    (1. - out_of_budget_soft) * (0.0) * (arg['param_cap'] - budgets_consumption)
            # print()
            # print(budget_reward_list[-1][0].item(), budgets_consumption[0].item())

            for i in range(LEN_eps_minus_one):
                performance_reward = torch.zeros_like(budget_reward_list[i])
                if i == LEN_eps_minus_one - 1:
                    if running_mean_loss00['mean_static_step'] == 0:
                        running_mean_loss00['mean_static'] = loss00.mean().item()
                        running_mean_loss00['mean_static_step'] += 1
                    elif running_mean_loss00['mean_static_step'] <= 200:
                        running_mean_loss00['mean_static'] = (running_mean_loss00['mean_static']  * (1.-1./ running_mean_loss00['mean_step']) 
                            + loss00.mean().item() *  1. / running_mean_loss00['mean_static_step'])
                        running_mean_loss00['mean_static_step'] += 1
                    else:
                        running_mean_loss00['mean_static']  = running_mean_loss00['mean_static']  * 0.995 + loss00.mean().item() * 0.005
                    performance_reward += - (loss00[..., None] / running_mean_loss00['mean_static'] )
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
        vf_coef = 0.5
        lam = 1.
        ent_coef = 0.0
        cliprange = 0.2
        out_range_coef = 1e-3
        nbatch = batchsize
        nbatch_train = 64
        assert nbatch % nbatch_train == 0
        noptepochs = 1 #2 #4
        # PPO 
        for _ in range(noptepochs):
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                ppo_sample_budget = sample_budget[start:end]
                ppo_a_dist_mean = []
                ppo_a_dist_stddev = []
                ppo_values = []
                # forward agent 
                layer_index, hidden, _ = static_actor.zeros_state(
                        nbatch_train, 
                        old_values.device, 
                        with_noise=False, 
                        is_train_gagent=None)
                for i in range(LEN_eps_minus_one):
                    _, (ppo_a_dist, ppo_v), _, layer_index, hidden = static_actor.predict_action(
                        budget=ppo_sample_budget,
                        layer_index=layer_index,
                        static_hidden=hidden,
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
                # entropy 
                entropy = torch.mean(ppo_action_dist.entropy())
                # action out of range penalty
                out_of_range = F.smooth_l1_loss(ppo_a_dist_mean, 
                            torch.clamp(ppo_a_dist_mean, action_range_min, action_range_max).detach())
                # Total loss
                actor_loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + out_range_coef * out_of_range
                # print(pg_loss, entropy, vf_loss)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(static_actor.parameters(), .5) #.5
                actor_optimizer.step()

        return actor_loss, rewards