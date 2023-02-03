import pytorch_lightning as pl
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MixtureSameFamily
from utils.stats import diagonal_mvn_log_prob


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(512, 384),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Dropout(p=0.1),
            nn.Linear(384, 256),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

    def forward(self, x):
        output = x.view(x.shape[0], -1)
        output = self.shared_trunk(output)
        return self.mu_head(output), self.logvar_head(output)


def make_decoder(latent_dim, input_dim):
    return nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.LeakyReLU(inplace=True, negative_slope=0.1),
        nn.Linear(256, 256),
        nn.LeakyReLU(inplace=True, negative_slope=0.1),
        nn.Linear(256, 384),
        nn.Dropout(p=0.1),
        nn.LeakyReLU(inplace=True, negative_slope=0.1),
        nn.Linear(384, 512),
        nn.LeakyReLU(inplace=True, negative_slope=0.1),
        nn.Linear(512, input_dim),
    )


class Model(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, n_components, lr):
        super().__init__()
        self.p_x_z_net = make_decoder(latent_dim, input_dim)
        self.q_z_x_net = Encoder(input_dim, latent_dim)
        self.logits_c = nn.Parameter(torch.ones(n_components) / n_components)
        self.mu_z_c = nn.Parameter(torch.zeros(n_components, latent_dim))
        self.logvar_z_c = nn.Parameter(torch.zeros(n_components, latent_dim))
        nn.init.xavier_normal_(self.mu_z_c)
        nn.init.xavier_normal_(self.logvar_z_c)
        self.lr = lr


    def sample_z(self, mu, var):
        if self.training:
            sd = var.sqrt()
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu


    def forward(self, x):
        # z ~ q(z|x)
        mu_tilde, logvar_tilde = self.q_z_x_net(x)
        var_tilde = F.softplus(logvar_tilde)
        z = self.sample_z(mu_tilde, var_tilde)
        # E_q(z,c|x)[log p(x|z)]
        mu_x = torch.sigmoid(self.p_x_z_net(z))
        log_p_x_z = (x * torch.log(mu_x) + (1 - x) * torch.log(1 - mu_x)).sum(-1).mean()
        # KL(q(z|x) || p(z))
        p_c = distributions.Categorical(logits=self.logits_c)
        p_z_c = distributions.Independent(distributions.Normal(loc=self.mu_z_c, scale=torch.exp(0.5 * self.logvar_z_c)), 1)
        p_z = MixtureSameFamily(p_c, p_z_c)
        log_q_z_x = diagonal_mvn_log_prob(z, mu_tilde, var_tilde, self.device)
        kl = (log_q_z_x - p_z.log_prob(z)).mean()
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