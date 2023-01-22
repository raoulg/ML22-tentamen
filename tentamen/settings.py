import numpy as np

from pathlib import Path
from typing import Union

from pydantic import BaseModel, HttpUrl
from ray import tune

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float

cwd = Path(__file__)
root = (cwd / "../..").resolve()


class Settings(BaseModel):
    datadir: Path
    testurl: HttpUrl
    trainurl: HttpUrl
    testfile: Path
    trainfile: Path
    modeldir: Path
    logdir: Path
    modelname: str
    batchsize: int


presets = Settings(
    datadir=root / "data/raw",
    testurl="https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Test_Arabic_Digit.txt",  # noqa N501
    trainurl="https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Train_Arabic_Digit.txt",  # noqa N501
    testfile=Path("ArabicTest.txt"),
    trainfile=Path("ArabicTrain.txt"),
    modeldir=root / "models",
    logdir=root / "logs",
    modelname="model.pt",
    batchsize=128,
)

presets_GRUAtt = Settings(
    datadir=root / "data/raw",
    testurl="https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Test_Arabic_Digit.txt",  # noqa N501
    trainurl="https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Train_Arabic_Digit.txt",  # noqa N501
    testfile=Path("ArabicTest.txt"),
    trainfile=Path("ArabicTrain.txt"),
    modeldir=root / "models",
    logdir=root / "logsGRUAtt",
    modelname="GRUAtt.pt",
    batchsize=32,
)

class BaseSearchSpace(BaseModel):
    input: int
    output: int
    tunedir: Path

    class Config:
        arbitrary_types_allowed = True


class LinearConfig(BaseSearchSpace):
    h1: int
    h2: int
    dropout: float


class GRUAttationConfig(BaseSearchSpace):
    hidden_size: int
    dropout: float
    num_layers: int
    num_heads: int
    


class LinearSearchSpace(BaseSearchSpace):
    h1: Union[int, SAMPLE_INT] = tune.randint(16, 128)
    h2: Union[int, SAMPLE_INT] = tune.randint(16, 128)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.5)

class GRUAttationSearchSpace(BaseSearchSpace):
    hidden_size: Union[int, SAMPLE_INT] = tune.qrandint(64, 256, 16)#tune.randint(64, 256)
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 32)
    # num_heads: Union[int, SAMPLE_INT] = tune.randint(2, 32)
    num_heads: Union[int, SAMPLE_INT] = tune.qrandint(2, 16, 4)
    # num_heads: Union[int, SAMPLE_INT] = tune.sample_from(lambda _: np.random.randint(2,32)) #tune.randint(2, 32)
    # hidden_size: Union[int, SAMPLE_INT] = tune.sample_from(lambda spec: spec.config.num_heads * np.random.randint(64,256))#tune.randint(64, 256)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.2)

