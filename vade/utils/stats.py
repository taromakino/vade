import torch

def make_gaussian(mu, logvar):
    '''
    The inputs must have shape (batch_size, dim). If we were to pass in a 1D array, it's ambiguous whether to return a
    batch of univariate Gaussians, or a single multivariate Gaussian.
    '''
    batch_size, dim = mu.shape
    mu, logvar = mu.squeeze(), logvar.squeeze()
    if dim == 1:
        dist = torch.distributions.Normal(loc=mu, scale=torch.exp(logvar / 2))
    else:
        cov_mat = torch.diag_embed(torch.exp(logvar), offset=0, dim1=-2, dim2=-1)
        dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov_mat)
    return dist