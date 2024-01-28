from abc import abstractmethod


class Function:
    @abstractmethod
    def __call__(self, *args):
        raise NotImplemented()

    @abstractmethod
    def derivative(self, *args):
        raise NotImplemented()

