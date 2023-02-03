from datetime import datetime

import torch
from loguru import logger

from tentamen.data import datasets
from tentamen.model import Accuracy
from tentamen.settings import presets_gruatt
from tentamen.train import trainloop

if __name__ == "__main__":
    logger.add(presets_gruatt.logdir / "01.log")

    trainstreamer, teststreamer = datasets.get_arabic(presets_gruatt)

    from tentamen.model import GRUAttention
    from tentamen.settings import GRUAttationConfig

    configs = [
        GRUAttationConfig(
            input=13,
            output=20,
            tunedir=presets_gruatt.logdir,
            hidden_size=160,
            num_layers=2,
            num_heads=2,
            dropout=0.02,
        ),
    ]

    for config in configs:
        model = GRUAttention(config.dict())  # type: ignore

        trainedmodel = trainloop(
            epochs=25,
            model=model,  # type: ignore
            optimizer=torch.optim.Adam,
            learning_rate=1e-3,
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics=[Accuracy()],
            train_dataloader=trainstreamer.stream(),
            test_dataloader=teststreamer.stream(),
            log_dir=presets_gruatt.logdir,
            train_steps=len(trainstreamer),
            eval_steps=len(teststreamer),
            patience=4,
            factor=0.5,
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = presets_gruatt.modeldir / (timestamp + presets_gruatt.modelname)
        logger.info(f"save model to {path}")
        torch.save(trainedmodel, path)
