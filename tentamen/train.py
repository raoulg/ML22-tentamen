from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from loguru import logger
from ray import tune
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tentamen.model import GenericModel


def trainbatches(
    model: GenericModel,
    traindatastreamer: Iterator,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    train_steps: int,
) -> float:
    model.train()
    train_loss: float = 0.0
    for _ in tqdm(range(train_steps), colour="#1e4706"):
        x, y = next(iter(traindatastreamer))
        optimizer.zero_grad()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().numpy()
    train_loss /= train_steps
    return train_loss


def evalbatches(
    model: GenericModel,
    testdatastreamer: Iterator,
    loss_fn: Callable,
    metrics: List,
    eval_steps: int,
) -> Tuple[Dict[str, float], float]:
    model.eval()
    test_loss: float = 0.0
    metric_dict: Dict[str, float] = {}
    for _ in range(eval_steps):
        x, y = next(iter(testdatastreamer))
        yhat = model(x)
        test_loss += loss_fn(yhat, y).detach().numpy()
        for m in metrics:
            metric_dict[str(m)] = (
                metric_dict.get(str(m), 0.0) + m(y, yhat).detach().numpy()
            )

    test_loss /= eval_steps
    for key in metric_dict:
        metric_dict[str(key)] = metric_dict[str(key)] / eval_steps
    return metric_dict, test_loss


def trainloop(
    epochs: int,
    model: GenericModel,
    optimizer: Callable,
    learning_rate: float,
    loss_fn: Callable,
    metrics: List,
    train_dataloader: Iterator,
    test_dataloader: Iterator,
    log_dir: Path,
    train_steps: int,
    eval_steps: int,
    patience: int = 10,
    factor: float = 0.9,
    tunewriter: bool = False,
) -> GenericModel:
    """

    Args:
        epochs (int) : Amount of runs through the dataset
        model: A generic model with a .train() and .eval() method
        optimizer : an uninitialized optimizer class. Eg optimizer=torch.optim.Adam
        learning_rate (float) : floating point start value for the optimizer
        loss_fn : A loss function
        metrics (List[Metric]) : A list of callable metrics.
            Assumed to have a __repr__ method implemented, see src.models.metrics
            for Metric details
        train_dataloader, test_dataloader (Iterator): data iterators
        log_dir (Path) : where to log stuff when not using the tunewriter
        train_steps, eval_steps (int) : amount of times the Iterators are called for a
            new batch of data.
        patience (int): used for the ReduceLROnPlatues scheduler. How many epochs to
            wait before dropping the learning rate.
        factor (float) : fraction to drop the learning rate with, after patience epochs
            without improvement in the loss.
        tunewriter (bool) : when running experiments manually, this should
            be False (default). If false, a subdir is created
            with a timestamp, and a SummaryWriter is invoked to write in
            that subdir for Tensorboard use.
            If True, the logging is left to the ray.tune.report


    Returns:
        _type_: _description_
    """

    optimizer_: torch.optim.Optimizer = optimizer(
        model.parameters(), lr=learning_rate
    )  # type: ignore

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_,
        factor=factor,
        patience=patience,
    )

    if not tunewriter:
        log_dir = dir_add_timestamp(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(epochs), colour="#1e4706"):
        train_loss = trainbatches(
            model, train_dataloader, loss_fn, optimizer_, train_steps
        )

        metric_dict, test_loss = evalbatches(
            model, test_dataloader, loss_fn, metrics, eval_steps
        )

        scheduler.step(test_loss)

        if tunewriter:
            tune.report(
                iterations=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                **metric_dict,
            )
        else:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            for m in metric_dict:
                writer.add_scalar(f"metric/{m}", metric_dict[m], epoch)
            lr = [group["lr"] for group in optimizer_.param_groups][0]
            writer.add_scalar("learning_rate", lr, epoch)
            metric_scores = [f"{v:.4f}" for v in metric_dict.values()]
            logger.info(
                f"Epoch {epoch} train {train_loss:.4f} test {test_loss:.4f} metric {metric_scores}"  # noqa E501
            )

    return model


def dir_add_timestamp(log_dir: Optional[Path] = None) -> Path:
    if log_dir is None:
        log_dir = Path(".")
    log_dir = Path(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = log_dir / timestamp
    logger.info(f"Logging to {log_dir}")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    return log_dir
