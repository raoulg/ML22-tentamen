import torch
from loguru import logger

from tentamen.data import datasets
from tentamen.model import Accuracy
from tentamen.settings import presets_gruatt
from tentamen.train import evalbatches

if __name__ == "__main__":
    logger.add(presets_gruatt.logdir / "01.log")

    trainstreamer, teststreamer = datasets.get_arabic(presets_gruatt)

    timestamp = "20230129-092207"
    path = presets_gruatt.modeldir / (timestamp + presets_gruatt.modelname)
    logger.info(f"loading model from {path}")
    model = torch.load(path)

    result = evalbatches(
        model=model,
        testdatastreamer=teststreamer.stream(),
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics=[Accuracy()],
        eval_steps=len(teststreamer),
    )

    logger.info(result)
