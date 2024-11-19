import abc

class sequence_generator:
    required_parameters : list[str]

    def __init__(self, parameters : dict[str|int]):
        for param_name in self.required_parameters:
            if param_name not in parameters:
                raise ValueError(f"required parameter {param_name} not in parameters")
        self.internal_parameters = {k:v for k,v in parameters.items() if k in self.required_parameters}
        self.initialize()

    @property
    def parameters(self) -> list[str]:
        return self.internal_parameters

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError("Abstract method called")

    @abc.abstractmethod
    def advance(self) -> dict[str,int]:
        raise NotImplementedError("Abstract method called")
