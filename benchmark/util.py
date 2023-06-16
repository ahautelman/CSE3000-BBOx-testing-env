import torch
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array


def get_random_guess(x_range: torch.Tensor, num_samples=1) -> torch.Tensor:
    """
    Return random torch.Tensor within x_range
    :param x_range: tuple of (min, max)
    :param num_samples:
    :return:
    """
    x_dim_ranges = (x_range[1] - x_range[0]) * torch.rand((num_samples, x_range.size(dim=1))) + x_range[0]
    return x_dim_ranges


def plot_function_with_selected_points(title, iter, X, Y, x_train, y_train, x_selected, y_selected):
    plt.figure(iter)
    with torch.no_grad():
        plt.plot(X.numpy(), Y.numpy(), label='Real function')
        plt.plot(x_train.numpy(), y_train.numpy(), 'k*', zorder=10, label='Observed points')
        plt.plot(x_selected.numpy(), y_selected.numpy(), 'r*', zorder=10, label='Selected point')
    plt.legend()
    plt.title(f"{title} Iteration {iter}")
    plt.show()


def plot_function_with_acq_values(title, iter, X, Y, x_train, y_train, x_selected, y_selected, acq_values):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    with torch.no_grad():
        ax1.set_title(f"{title} Iteration {iter}")
        ax1.plot(X.numpy(), Y.numpy(), label='Real function')
        ax1.plot(x_train.numpy(), y_train.numpy(), 'k*', zorder=10, label='Observed points')
        ax1.plot(x_selected.numpy(), y_selected.numpy(), 'r*', zorder=10, label='Selected point')

        ax2.set_title('Acquisition function')
        ax2.plot(X.numpy(), acq_values, label='Acquisition values')
    fig.tight_layout()
    plt.show()


def jax_array_to_tensor(x: Array) -> torch.Tensor:
    # return torch.from_numpy(np.asarray(x))
    return torch.tensor(np.asarray(x)).double()
    # return torch.tensor(np.asarray(x))

def tensor_to_jax_array(x: torch.Tensor) -> Array:
    return jnp.array(x.detach().numpy())


def constrain(x: torch.Tensor, x_range: torch.Tensor) -> torch.Tensor:
    """
    Constrain x to x_range
    :param x: torch.Tensor
    :param x_range: torch.Tensor
    :return: torch.Tensor
    """
    with torch.no_grad():
        return x.clamp_(x_range[0], x_range[1])
