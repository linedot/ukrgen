from dataclasses import dataclass

from .ukr import (
        ukr_composition,
        sto_description,
        block_description,
        dimension
    )

class gemm_composition(ukr_composition):

    component_reference_map = {
        "alpha" : "C",
        "beta" : "C"
    }

    @classmethod
    def get_components(cls) -> list[str]:
        return ["A","B","AB","C","alpha","beta"]

    @classmethod
    def get_parameterized_components(cls) -> list[str]:
        return ["A","B","AB","C"]


    @classmethod
    def get_sto_descriptions(cls) -> list[sto_description]:

        dim=dimension

        return [
            sto_description(
                name="mm",
                generator="mm",
                components=["A","B","AB"],
                component_references={},
                component_sup_tiles={
                    "A" : {"a","b"},
                    "B" : {"a","b"},
                    "AB" : {"c"},
                    },
                dimensions = {
                    "A" : (dim("m"),dim("k")),
                    "B" : (dim("k"),dim("n")),
                    "AB" : (dim("m"),dim("n"))
                },
                preload=True,
                tail=True),
            sto_description(
                name="betascale",
                generator="mm",
                components=["C","beta","C"],
                component_references={"beta":"C"},
                component_sup_tiles={
                    "C" : {"a","b","c"},
                    "beta" : {"a","b"},
                    },
                dimensions = {
                    "C" : (dim("m"),dim("n")),
                    "beta" : (
                        dim(lambda ctx : ctx.resolved_components["C"][1]),
                        dim(lambda ctx : ctx.resolved_components["C"][1])
                    )
                },
                preload=False,
                tail=False,
                bands = (0,0)),
            sto_description(
                name="alphascale",
                generator="mm",
                components=["AB","alpha","C"],
                component_references={"alpha":"C"},
                component_sup_tiles={
                    "C" : {"c"},
                    "alpha" : {"a","b"},
                    "AB" : {"a","b"},
                    },
                dimensions = {
                    "AB" : (dim("m"),dim("n")),
                    "C" : (dim("m"),dim("n")),
                    "alpha" : (
                        dim(lambda ctx : ctx.resolved_components["C"][1]),
                        dim(lambda ctx : ctx.resolved_components["C"][1])
                    )
                },
                preload=False,
                tail=False,
                bands = (0,0)),
            sto_description(
                name="store",
                generator="store",
                components=["C"],
                component_references={},
                component_sup_tiles={ "C" : {"c"}},
                dimensions = {
                    "C" : (dim("m"),dim("n")),
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
