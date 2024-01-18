from abc import ABC, abstractmethod


class Backend(ABC):
    @abstractmethod
    def exec(self, verbose):
        pass
