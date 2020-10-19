import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianActorCritic(nn.Module):
    def __init__(self, rnn_hidden_size, prunable_layers_n_channels, sigma=0.05):
        super(GaussianActorCritic, self).__init__()
        self.prunable_layers_n_channels = prunable_layers_n_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.sigma = sigma

        self.layer_index_size = len(prunable_layers_n_channels)
        self.encoder_size = self.rnn_hidden_size # could be different
        self.rnn_input_size = self.encoder_size + 1 # 1 for budget
        self.rnn = nn.RNN(self.rnn_input_size, self.rnn_hidden_size, num_layers=1)


        self.encoders = []

        for in_channels in self.prunable_layers_n_channels:
            self.encoders.append(self.create_encoder(in_channels))

        self.encoders = nn.ModuleList(self.encoders)

        self.decoder_mu = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, 1),
        )
        self.decoder_log_sigma = nn.Linear(self.rnn_hidden_size, 1)
        self.decoder_value = nn.Linear(self.rnn_hidden_size , 1)


        self.decoder_mu[0].bias.data.fill_(0.)
        self.decoder_mu[0].weight.data *= 0.2
        self.decoder_log_sigma.bias.data.fill_(-4.)


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
            hidden = torch.zeros(self.rnn.num_layers, 
                            batch_size, self.rnn_hidden_size, device=device)
            layer_index = 0
        return layer_index, hidden, None

    def predict_action(self, state, budget, layer_index, hidden, with_noise=True, is_train_gagent=None):
        
        x = state
        x = self.encoders[layer_index](state)
        x = torch.cat((x, budget.detach()), dim=-1)
        x, hidden = self.rnn(x.unsqueeze(0), 
                                hidden)
        x = x.squeeze(0)
        action_mu = self.decoder_mu(x)
        action_mu = action_mu  + budget
        action_log_sigma = self.decoder_log_sigma(x)
        value = self.decoder_value(x)
        gaussian_dist = torch.distributions.normal.Normal(action_mu, 
                scale=torch.exp(action_log_sigma)
                )
        if with_noise:
            action = gaussian_dist.rsample().detach()
        else:
            action = action_mu.detach()

        with torch.no_grad():
            layer_index = layer_index + 1
        return action, (gaussian_dist, value), budget, layer_index, hidden
        

    def forward():
        raise NotImplementedError

