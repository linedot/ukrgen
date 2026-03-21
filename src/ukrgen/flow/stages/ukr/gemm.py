from dataclasses import dataclass

from .ukr import ukr_composition,sto_description,block_description

class gemm_composition(ukr_composition):

    component_reference_map = {
        "alpha" : "C",
        "beta" : "C"
    }

    @classmethod
    def get_components(cls) -> list[str]:
        return ["A","B","AB","C","alpha","beta"]

    @classmethod
    def get_component_sup_tiles(cls, component : str) -> set[str]:
        if component in {"A"}:
            return {"a","b"}
        elif component in {"B"}:
            return {"a","b"}
        elif component in ["AB","C","alpha","beta"]:
            return {"c"}

        raise ValueError(f"Invalid component {component}")

    @classmethod
    def get_parameterized_components(cls) -> list[str]:
        return ["A","B","AB","C"]

    @classmethod
    def get_component_reference(cls, component: str) -> str:
        if component in cls.component_reference_map:
            return cls.component_reference_map[component]
        
        return component

    @classmethod
    def get_sto_descriptions(cls) -> list[sto_description]:
        return [
            sto_description(
                name="mm",
                generator="mm",
                components=["A","B","AB"],
                dimensions = {
                    "A" : ("m","k"),
                    "B" : ("k","n"),
                    "AB" : ("m","n")
                },
                preload=True,
                tail=True),
            sto_description(
                name="betascale",
                generator="mm",
                components=["C","beta","C"],
                dimensions = {
                    "C" : ("m","n"),
                    "beta" : ("n","n"),
                },
                preload=False,
                tail=False,
                bands = (0,0)),
            sto_description(
                name="alphascale",
                generator="mm",
                components=["AB","alpha","C"],
                dimensions = {
                    "AB" : ("m","n"),
                    "C" : ("m","n"),
                    "alpha" : ("n","n"),
                },
                preload=False,
                tail=False,
                bands = (0,0)),
            sto_description(
                name="store",
                generator="store",
                components=["C"],
                dimensions = {
                    "C" : ("m","n"),
                },
                preload=False,
                tail=False),
            
        ]

    @classmethod
    def get_blocks(cls) -> list[block_description]:
        return [
            block_description(loop=True,
                              stos=["mm"]),
            block_description(loop=False,
                              stos=["betascale",
                                    "alphascale",
                                    "store"]),
        ]
