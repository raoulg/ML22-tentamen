from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from tentamen.data import data_tools
from tentamen.settings import Settings


def get_arabic(presets: Settings) -> Tuple[BaseDatastreamer, BaseDatastreamer]:

    data_tools.get_file(
        data_dir=presets.datadir,
        filename=presets.trainfile,
        url=presets.trainurl,
        unzip=False,
    )

    data_tools.get_file(
        data_dir=presets.datadir,
        filename=presets.testfile,
        url=presets.testurl,
        unzip=False,
    )

    trainpath = presets.datadir / presets.trainfile
    testpath = presets.datadir / presets.testfile
    logger.info(f"Loading data from {trainpath} and {testpath}")
    traindataset = ArabicDataset(trainpath)
    testdataset = ArabicDataset(testpath)

    trainstreamer = BaseDatastreamer(
        dataset=traindataset,
        batchsize=presets.batchsize,
        preprocessor=preprocessor,
    )

    teststreamer = BaseDatastreamer(
        dataset=testdataset,
        batchsize=presets.batchsize,
        preprocessor=preprocessor,
    )

    logger.info("Returning trainstreamer, teststreamer")
    return trainstreamer, teststreamer


class BaseDataset:
    """The main responsibility of the Dataset class is to load the data from disk
    and to offer a __len__ method and a __getitem__ method
    """

    def __init__(self, file: Path) -> None:
        self.file: Path = file
        self.data: List = []
        self.labels: List = []
        self.process_data()

    def process_data(self) -> None:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx], self.labels[idx]


class ArabicDataset(BaseDataset):
    def process_data(self) -> None:
        with open(self.file, "r") as f:
            data = f.read()
        # split on a whiteline, optional additional spaces on the whiteline
        lines = re.split(r"\n *\n", data)
        for x in lines:
            # remove the '\n' linebreaks
            # because np.fromstring does not read matrices
            arr_ = re.sub(r"\n", " ", x.strip())
            # read data as an array
            arr = np.fromstring(arr_, dtype=float, sep=" ")
            # we know the matrices are sized (seq,13)
            # float32 because pytorch modules are float32
            self.data.append(torch.tensor(arr.reshape((-1, 13)), dtype=torch.float32))
        self.labels = self.get_labels()

    def get_labels(self) -> List[str]:
        if re.search(r"Train", self.file.name):
            blocksize = 660
        else:
            blocksize = 220
        # there are 660 x 10 digits
        d = np.repeat(range(10), blocksize)
        # there are 330 males and 330 females, 10 times.
        g = np.tile(np.repeat(["m", "f"], blocksize // 2), 10)
        labels = ["".join([str(a), str(b)]) for a, b in zip(d, g)]
        assert len(labels) == len(self.data)
        return labels


class BaseDatastreamer:
    """This datastreamer wil never stop
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(
        self,
        dataset: BaseDataset,
        batchsize: int,
        preprocessor: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.batchsize = batchsize
        self.preprocessor = preprocessor
        self.size = len(self.dataset)
        self.reset_index()

    def __len__(self) -> int:
        return int(len(self.dataset) / self.batchsize)

    def reset_index(self) -> None:
        self.index_list = np.random.permutation(self.size)
        self.index = 0

    def batchloop(self) -> Sequence[Tuple]:
        batch = []
        for _ in range(self.batchsize):
            x, y = self.dataset[int(self.index_list[self.index])]
            batch.append((x, y))
            self.index += 1
        return batch

    def stream(self) -> Iterator:
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            if self.preprocessor is not None:
                X, Y = self.preprocessor(batch)  # noqa N806
            else:
                X, Y = zip(*batch)  # noqa N806
            yield X, Y


mapping = {
    k: i for i, k in enumerate([str(i) + g for g in ["m", "f"] for i in range(10)])
}


def preprocessor(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    X, y_ = zip(*batch)  # noqa: N806
    X = pad_sequence(X, batch_first=True)  # noqa: N806
    y = torch.tensor([mapping[label] for label in y_])
    return X, y
