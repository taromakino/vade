import os
import pytorch_lightning as pl
import torch
import torch.distributions as distributions
import torch.nn as nn
from torch.optim import Adam
from utils.nn_utils import MLP, MixtureSameFamily
from utils.stats import make_gaussian


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.mu_net = MLP(input_dim, hidden_dims, output_dim)
        self.logvar_net = MLP(input_dim, hidden_dims, output_dim)


    def forward(self, *args):
        return self.mu_net(*args), self.logvar_net(*args)


class Model(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_components, lr):
        super().__init__()
        self.p_x_z_net = MLP(latent_dim, hidden_dims, input_dim)
        self.q_z_x_net = GaussianMLP(input_dim, hidden_dims, latent_dim)
        self.logits_c = nn.Parameter(torch.ones(n_components) / n_components)
        self.mu_z_c = nn.Parameter(torch.zeros(latent_dim, n_components))
        self.logvar_z_c = nn.Parameter(torch.zeros(latent_dim, n_components))
        nn.init.xavier_normal_(self.mu_z_c)
        nn.init.xavier_normal_(self.logvar_z_c)
        self.lr = lr

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu


    def forward(self, x):
        # z ~ q(z|x)
        mu_tilde, logvar_tilde = self.q_z_x_net(x)
        z = self.sample_z(mu_tilde, logvar_tilde)
        # E_q(z,c|x)[log p(x|z)]
        mu_x = torch.sigmoid(self.p_x_z_net(z))
        log_p_x_z = (x * torch.log(mu_x) + (1 - x) * torch.log(1 - mu_x)).sum(-1).mean()
        # KL(q(z|x) || p(z))
        q_z_x = make_gaussian(mu_tilde, logvar_tilde)
        p_c = distributions.Categorical(logits=self.logits_c)
        p_z_c = distributions.Independent(distributions.Normal(loc=self.mu_z_c.permute(1, 0), scale=torch.exp(0.5 *
            self.logvar_z_c).permute(1, 0)), 1)
        p_z = MixtureSameFamily(p_c, p_z_c)
        kl = (q_z_x.log_prob(z) - p_z.log_prob(z)).mean()
        elbo = log_p_x_z - kl
        return {
            "loss": -elbo,
            "kl": kl
        }


    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        return out["loss"]


    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True)
        self.log("val_kl", out["kl"], on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        self.log("test_loss", out["loss"], on_step=False, on_epoch=True)
        self.log("test_kl", out["kl"], on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)