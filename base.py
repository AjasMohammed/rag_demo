from abc import ABC, abstractmethod


class BaseRAGDBClient(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def fetch_named_data(self, query: str) -> list[dict]:
        pass
