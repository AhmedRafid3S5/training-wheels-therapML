import torch
from torch import Tensor
from jaxtyping import Float  # Assuming the signature uses jaxtyping for shape annotation

def myLayerNorm(   
    input: Float[Tensor, "batch ..."],
    gamma: Float[Tensor, "batch ..."],
    beta: Float[Tensor, "batch ..."],):
    """Provided a tensor input and learned gamma and beta across each dim, normalize the
       inputs for stable learning"""
    #normalize across all dimensions except for the batch dimension
    eps = 1e-5
    dims = tuple(range(1, input.ndim))

    #find batch mean and variance
    #NOTE: unbiased=False makes use of the population variance instead of the sample variance
    #NOTE: keepdim set to True to ensure proper broadcasting of mean and variance
    mu  = input.mean(dim=dims,keepdim=True)
    var = input.var(dim=dims,unbiased=False,keepdim=True)

    #standardize the inputs
    x_hat = (input - mu) / torch.sqrt(var + eps) #use eps to avoid div by 0 error when var == 0

    #apply affine transformtion parameters
    y_hat = gamma*x_hat + beta

    return y_hat  

def myRMSnorm( 
    input: Float[Tensor, "batch ..."],
    gamma: Float[Tensor, "batch ..."],):

    eps = 1e-5
    #NOTE: Must compute squared_mean across dimensions
    squared_mean = torch.mean(torch.square(input),dim=-1,keepdim=True)
    rms = torch.sqrt(squared_mean + eps)

    return (input/ rms)*gamma
