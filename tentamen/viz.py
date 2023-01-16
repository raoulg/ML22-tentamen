from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def show_reconstruction(y: np.ndarray, yhat: np.ndarray, filepath: Path) -> None:
    plt.plot(y, "b")
    plt.plot(yhat, "r")
    plt.fill_between(np.arange(140), yhat, y, color="lightcoral")
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.savefig(filepath)
    logger.success(f"saved grid to {filepath}")
