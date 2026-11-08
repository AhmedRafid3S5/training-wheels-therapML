import torch
from torch import Tensor
from jaxtyping import Float  # Assuming the signature uses jaxtyping for shape annotation

def inverted_dropout(input: Float[Tensor, "..."], prob: float):
    """
    Applies inverted dropout to the input tensor.
    prob: The probability of dropping an element (0.0 to 1.0).
    """

    #if probability of dropout is 0.0, all neurons are active
    #just return the input
    if prob == 0.0:
        return input
    if prob == 1.0:
        return torch.zeros_like(input)

    keep_prob = 1.0 - prob

    #NOTE: must consider cases where keep_prob = 0.0, avoid div  by 0.0 error!!
    #generate the bernoulli mask
    #generate a tensor of random numbers
    #if tensor element >= prob, set to 1, else set to 0
    mask = (torch.rand_like(input) > prob).to(input.dtype)
    #       ~~~~~~~~~~^~~~~~~~~~~~ creates the random numbers btwn [0,1)
    scaled_mask = mask / keep_prob
    return input*scaled_mask