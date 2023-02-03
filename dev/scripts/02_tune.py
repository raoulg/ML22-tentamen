from typing import Dict

import ray
import torch
from filelock import FileLock
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

from tentamen.data import datasets
from tentamen.model import Accuracy, Linear, GRUAttention
from tentamen.settings import LinearSearchSpace, GRUAttationSearchSpace #, presets
from tentamen.settings import presets_GRUAtt as presets
from tentamen.train import trainloop


def train(config: Dict) -> None:
    datadir = presets.datadir

    with FileLock(datadir / ".lock"):
        trainstreamer, teststreamer = datasets.get_arabic(presets)

    model = GRUAttention(config)  # type: ignore

    trainloop(
        epochs=50,
        model=model,  # type: ignore
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics=[Accuracy()],
        train_dataloader=trainstreamer.stream(),
        test_dataloader=teststreamer.stream(),
        log_dir=presets.logdir,
        train_steps=len(trainstreamer),
        eval_steps=len(teststreamer),
        tunewriter=True,
        patience=3,
        factor=0.5,
    )


if __name__ == "__main__":
    ray.init()

    config = GRUAttationSearchSpace(
        input=13,
        output=20,
        tunedir=presets.logdir,
    )

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=5,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()

    analysis = tune.run(
        train,
        config=config.dict(),
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        local_dir=config.tunedir,
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()


# def train(config: Dict) -> None:
#     datadir = presets.datadir

#     with FileLock(datadir / ".lock"):
#         trainstreamer, teststreamer = datasets.get_arabic(presets)

#     model = Linear(config)  # type: ignore

#     trainloop(
#         epochs=30,
#         model=model,  # type: ignore
#         optimizer=torch.optim.Adam,
#         learning_rate=1e-3,
#         loss_fn=torch.nn.CrossEntropyLoss(),
#         metrics=[Accuracy()],
#         train_dataloader=trainstreamer.stream(),
#         test_dataloader=teststreamer.stream(),
#         log_dir=presets.logdir,
#         train_steps=len(trainstreamer),
#         eval_steps=len(teststreamer),
#         tunewriter=True,
#     )


# if __name__ == "__main__":
#     ray.init()

#     config = LinearSearchSpace(
#         input=13,
#         output=20,
#         tunedir=presets.logdir,
#     )

#     reporter = CLIReporter()
#     reporter.add_metric_column("Accuracy")

#     bohb_hyperband = HyperBandForBOHB(
#         time_attr="training_iteration",
#         max_t=30,
#         reduction_factor=3,
#         stop_last_trials=False,
#     )

#     bohb_search = TuneBOHB()

#     analysis = tune.run(
#         train,
#         config=config.dict(),
#         metric="test_loss",
#         mode="min",
#         progress_reporter=reporter,
#         local_dir=config.tunedir,
#         num_samples=20,
#         search_alg=bohb_search,
#         scheduler=bohb_hyperband,
#         verbose=1,
#     )

#     ray.shutdown()
