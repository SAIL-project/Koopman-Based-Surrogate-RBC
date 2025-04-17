import logging
from typing import Tuple

from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from torch import Tensor

from koopman_rbc.data.rbc_dataset import RBCField


def plot_field_sequence(seq: Tensor, field: RBCField) -> animation.Animation:
    # Set animation logger level
    logger = logging.getLogger("matplotlib.animation")
    logger.setLevel(logging.ERROR)

    # Draw artists
    artists = []
    steps = seq.shape[0]
    for i in range(steps):
        fig, ax, im = plot_field(seq[i], field)
        artists.append([im])
    return animation.ArtistAnimation(fig, artists, blit=True)


def plot_field(x: Tensor, field: RBCField) -> Tuple[Figure, Axes, AxesImage]:
    fig, ax = plt.subplots()
    ax.set_axis_off()
    if field == RBCField.T:
        vmin, vmax = 1, 2
    else:
        vmin, vmax = None, None

    im = ax.imshow(x[field], cmap="coolwarm", vmin=vmin, vmax=vmax)

    return fig, ax, im


def plot_difference(diff: Tensor) -> Figure:
    fig, ax = plt.subplots()
    ax.set_axis_off()
    im = ax.imshow(diff, cmap="binary")
    return fig, ax, im
