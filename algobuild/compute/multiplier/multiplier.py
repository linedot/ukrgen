import abc

class multiplier:

    @abc.abstractmethod
    def process(self):
        raise NotImplementedError("Attempting to call base multiplier.process() method")

