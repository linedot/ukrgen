# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import sys

import logging
from dataclasses import dataclass
from enum import Enum,auto
from typing import Iterator
import itertools

from .stage import stage
from .unvec import unvec_stage
from .ukr import ukr_composition_map,ukr_composition
from .ukr.ukr import dimension_resolution_context,dimension
from ..ukr_context import ukr_context
from ..stage_param import stage_param
from ...specializers.asm import op_support
from ...components.tile import dimension_properties,tile


class geometry_transformation(Enum):
    none = auto()
    transpose = auto()

@dataclass
class geometry_component:
    name : str
    transformation : geometry_transformation

@dataclass
class geometry_generation_pass:
    order : tuple[int,int,int]
    transformation : geometry_transformation


def generate_component_geometry(
        components : set[str],
        sup_tile_components : dict[str,set[str]],
        generator_name : str
        ) -> Iterator[list[geometry_component]]:

    passes = [
        geometry_generation_pass(
            order=(0,1,2),
            transformation=geometry_transformation.none)
    ]
    if generator_name in {"mm","gemm"}:
        passes.append(
            geometry_generation_pass(
                order=(1,0,2),
                transformation=geometry_transformation.transpose)
        )

    a_components = sup_tile_components['a'].intersection(components)
    b_components = sup_tile_components['b'].intersection(components)
    c_components = sup_tile_components['c'].intersection(components)

    print(f"a-tile comps: {a_components}")
    print(f"b-tile comps: {b_components}")
    print(f"c-tile comps: {c_components}")
     
    for genpass in passes:
        for a,b,c in itertools.product(
                a_components,
                b_components,
                c_components
                ):
            
            component_tuple = (a, b, c)

            yield [
                geometry_component(name=component_tuple[i],
                                   transformation=genpass.transformation)
                for i in genpass.order
            ]


def mapping_valid(sto_dimensions : dict[str,tuple[str,str]],
                  geometry: list[geometry_component],
                  mapping : dict[str,tile],
                  sup : op_support) -> bool:


    print(mapping)
    geometry_tiles = []
    for gc in geometry:

        da,db = sto_dimensions[gc.name]
        t = tile(dima=mapping[da],dimb=mapping[db])
        print(f"component: {gc.name} - {t}")
        print("----------------------------")
        geometry_tiles.append(t)


    print(f"sup            : {sup}")
    print(f"resulting tiles: {geometry_tiles}")

    tile_matches = [gt.dima == st.dima and \
           gt.dimb == st.dimb for \
           gt,st in zip(geometry_tiles, 
                        (sup.a_tile,sup.b_tile,sup.c_tile))]

    print(tile_matches)

    if all(tile_matches):
        return True

    return False

def resolve_dims_with_geometry(
        dim_ctx : dimension_resolution_context,
        dims : dict[str,tuple[dimension,dimension]],
        geometry : list[geometry_component]):

    for gc in geometry:
        da,db = dims[gc.name]

        if any(d.is_dynamic for d in (da,db)):
            continue
        
        if gc.name not in dim_ctx.resolved_components:
            if gc.transformation == geometry_transformation.transpose:
                dim_ctx.resolved_components[gc.name] = (str(db),str(da))
            else:
                dim_ctx.resolved_components[gc.name] = (str(da),str(db))
        

    for gc in geometry:
        da,db = dims[gc.name]

        if all(not d.is_dynamic for d in (da,db)):
            continue

        da_str = da.resolve(dim_ctx)
        db_str = db.resolve(dim_ctx)

        dim_ctx.resolved_components[gc.name] = (da_str,db_str)


class geometry_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)
        
        self.debug = logging.getLogger("GEOMETRY").debug
        self.error = logging.getLogger("GEOMETRY").error

        ukr = self.context.params["ukr"].value

        composition = ukr_composition_map[ukr]

        sup = self.context.sup
        dps = {sup.a_tile.dima, sup.a_tile.dimb,
               sup.b_tile.dima, sup.a_tile.dimb, 
               sup.c_tile.dima, sup.b_tile.dimb}


        global_dims = set()

        # Gather all static dimensions
        for sto in composition.get_sto_descriptions():
            for dims in sto.dimensions.values():
                for d in dims:
                    if not d.is_dynamic:
                        global_dims.add(str(d))

        # Lock in order
        global_dims = list(global_dims)

        valid_global_solutions = []

        # Go through all possible assignments
        for assignment in itertools.product(dps, repeat=len(global_dims)):
            global_mapping = dict(zip(global_dims, assignment))

            dim_ctx = dimension_resolution_context(
                    global_mapping=global_mapping,
                    sup=sup)


            is_valid_global_solution=True
            valid_local_solutions = {}

            print(f"Checking global mapping {global_mapping}")


            #Check all STOs
            for sto in composition.get_sto_descriptions():

                print(f"Checking STO {sto.name}")

                valid_geometries = []

                for geometry in generate_component_geometry(
                        sto.components, 
                        composition.get_sup_tile_components(),
                        sto.generator):
                    print(f"checking {geometry}")

                    resolve_dims_with_geometry(
                            dim_ctx,
                            sto.dimensions,
                            geometry
                            )

                    if mapping_valid(dim_ctx.resolved_components,
                                     geometry,
                                     global_mapping, 
                                     sup):
                        print(f"adding {geometry}")
                        valid_geometries.append(geometry)

                if not valid_geometries:
                    is_valid_global_solution = False
                    print(f"no valid global solutions")
                    break

                valid_local_solutions[sto.name] = valid_geometries

            if is_valid_global_solution:
                valid_global_solutions.append({
                    "mapping" : global_mapping,
                    "geometries" : solution_geometries
                    })

        if not valid_global_solutions:
            self.error("FATAL: No valid geometries found for hardware signature.")
            sys.exit(-1)



        print(f"Possible geometry solutions:")
        for i,solution in enumerate(valid_global_solutions):
            mapping = solution["mapping"]
            geometries = solution["geometries"]
            print(f"Solution {i}:")
            print(f"  mapping: {mapping}:")
            for j,g in enumerate(geometries):
                print(f"    Geometry {i}:")
                print(f"    {g.name} - {g.transformation}")
                        
        sys.exit(-1)



    def progress(self) -> list[stage]:

        self.context.params.update(self.params)


        vecdir = self.context.params["vecdir"].value
        assert vecdir in ["M","N"], f"Invalid vecdir: {vecdir}"


        # TODO: There should be some kind of generalization
        # TODO: real/data tile are already used in some places,
        #       maybe there should be support/component tiles?
        b_map = {"M" : -1, "N" : 1}
        a_map = {"M" : 0, "N" : -1}
        c_map = {"M" : 0, "N" : 1}

        if vecdir == "N":
            self.context.sup.b_tile,self.context.sup.a_tile = \
              self.context.sup.a_tile,self.context.sup.b_tile
    
        if -1 != b_map[vecdir]:
            self.context.sup.b_tile = copy_with_vecdir(
                    t=self.context.sup.b_tile,
                    vectorized_dimension=b_map[vecdir])
        if -1 != a_map[vecdir]:
            self.context.sup.a_tile = copy_with_vecdir(
                    t=self.context.sup.a_tile,
                    vectorized_dimension=a_map[vecdir])
        if -1 != c_map[vecdir]:
            self.context.sup.c_tile = copy_with_vecdir(
                    t=self.context.sup.c_tile,
                    vectorized_dimension=c_map[vecdir])

        # Do we need to unvec?
        if self.context.params["op"].value == "fma" and \
                self.context.sup.b_tile.is_vector and \
                self.context.sup.a_tile.is_vector:

            return [unvec_stage]
        else:
            return list()
