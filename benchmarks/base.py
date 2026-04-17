from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader


class BenchmarkDataset(Dataset, ABC):

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx): ...

    @abstractmethod
    def augment(self, sample):
        ...


class Benchmark(ABC):

    @abstractmethod
    def get_train_loader(self, batch_size: int) -> DataLoader: ...

    @abstractmethod
    def get_test_loader(self, batch_size: int) -> DataLoader: ...

    @abstractmethod
    def evaluate(self, model, device) -> dict: ...