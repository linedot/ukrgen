from abc import ABC,abstractclassmethod
from dataclasses import dataclass
from typing import Callable


from ....specializers.asm import op_support

class dimension_resolution_context:
    def __init__(self,
                 global_mapping : dict[str, str],
                 sup : op_support):
        self.global_mapping = global_mapping
        self.sup = sup
        self.resolved_components : dict[str, tuple[str,str]] = {}

class dimension:
    def __init__(self,
            definition: str|Callable[[dimension_resolution_context], str]):
        self.definition = definition

    @property
    def is_dynamic(self):
        return callable(self.definition)


    def resolve(self, ctx: dimension_resolution_context):
        if self.is_dynamic:
            return self.definition(ctx)

        return self.definition

    def __str__(self):
        if self.is_dynamic:
            return "<dynamic>"

        return self.definition

    def __repr__(self):
        return str(self)


@dataclass
class sto_description:
    name : str
    generator : str
    components : list[str]
    component_references : dict[str,str]
    component_sup_tiles : dict[str,set[str]]
    dimensions : dict[str,tuple[dimension,dimension]]
    preload : bool
    tail : bool
    op_override : str|None = None
    bands : tuple[int,int] = (-1,-1)

    def get_component_reference(self, component: str) -> str:
        """
        If the component parameters like data type are referenced from another
        component - return that component - otherwise returns the original name
        """
        if component in self.component_references:
            return self.component_references[component]
        
        return component

    def get_sup_tile_components(self) -> dict[str,set[str]]:
        """
        Returns a map of sup tile names to possible components for this STO
        """
        # No longer hardcoded sup tile names. We gather them from the definitions
        #sup_tile_components = {
        #    "a" : set(),
        #    "b" : set(),
        #    "c" : set(),
        #}

        sup_tile_components = {}
        for c in self.components:
            sup_tiles = self.component_sup_tiles[c]
            for stile in sup_tiles:
                if not stile in sup_tile_components:
                    sup_tile_components[stile] = set()
                sup_tile_components[stile].add(c)

        return sup_tile_components

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

