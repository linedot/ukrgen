from abc import ABC,abstractclassmethod
from dataclasses import dataclass

@dataclass
class sto_description:
    name : str
    generator : str
    components : list[str]
    dimensions : dict[str,tuple[str,str]]
    preload : bool
    tail : bool
    op_override : str|None = None
    bands : tuple[int,int] = (-1,-1)

@dataclass
class block_description:
    loop : bool
    stos : list[str]

class ukr_composition(ABC):


    @abstractclassmethod
    def get_components(cls) -> list[str]:
        """
        Returns all components the microkernel is using
        """

    @abstractclassmethod
    def get_component_sup_tiles(cls) -> set[str]:
        """
        Returns the ISA support triple tile this component can be mapped onto, i.e
        "a","b" or "c"
        """

    @abstractclassmethod
    def get_parameterized_components(cls) -> list[str]:
        """
        returns components that allow parameterization, like number
        of registers to allocate or how many registers to preload, etc...
        """


    @abstractclassmethod
    def get_sto_descriptions(cls) -> list[sto_description]:
        """
        returns STO descriptions of all STOs of the microkernel
        """

    @abstractclassmethod
    def get_blocks(cls) -> list[block_description]:
        """
        returns block descriptions of all blocks of the microkernel
        """
