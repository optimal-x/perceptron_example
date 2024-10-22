import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animate(epochs: int, losses):
    fig, ax = plt.subplots()
    ax.set_xlim(0, epochs)
    ax.set_xlim(0, epochs)
    (line,) = ax.plot([], [], lw=2)
    ax.set_title("Training Loss Ove Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    def init(*args):
        line.set_data([], [])
        return line

    def update(*frame):
        x = np.arange(0, frame)
        y = losses[:frame]
        line.set_data(x, y)
        return line

    animation = FuncAnimation(
        fig, update, frames=range(1, epochs + 1), init_func=init, blit=True
    )
    plt.show()


def plot(epochs, losses):
    x = np.arange(0, epochs)
    y = losses
    plt.plot(x, y)
    plt.show()
