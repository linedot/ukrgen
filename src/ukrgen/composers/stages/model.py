# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from copy import deepcopy

from .composition import composition_stage
from .irmod import unvec_lsc_stage
from ..gemm import gemm_context
from ..stage_param import stage_param
from ...components.tile import dimension_properties,dimension_type,scalar_tile
from ...models.offset_mapper import strided_mapper,same_address_mapper
from ...models.addr_parameters import calculate_addr_parameters
from ...models.load_store_cpu import load_store_cpu
from ...models.addr_resolver import addr_resolver

class lsc_model_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        # TODO: Make this and all following stages indepent of "ukr"
        #       Perhaps components should be set somewhere else?
        components = ["A","B","C"]
        if context.params["ukr"].value == "gemm":
            components = ["A", "B", "AB", "C"]
        

        # NOTE: On the other hand columns and rows are kind of matmul-specific...
        for cr in ["column","row"]:

            self.params[f"{cr}-strides"] = stage_param(
                    value=[],
                    default=[],
                    description=f"Components with general {cr} strides",
                    required=False)



        multiareg_strategies = ["interleave","split","phase"]

        default_mars = {
            "A" : "interleave",
            "B" : "interleave",
            "AB" : "phase",
            "C" : "phase",
        }
        for c in components:
            self.params[f"{c}-data-regs"] = stage_param(
                    value=None,
                    description=f"Number of data registers to use for component {c}")

            self.params[f"{c}-addr-regs"] = stage_param(
                    value=1, 
                    description=f"Number of address register to use for component {c}",
                    default=1)

            self.params[f"{c}-multiaddr-strat"] = stage_param(
                    value=default_mars[c], 
                    description=f"Strategy for using multiple address registers for component {c}",
                    default=default_mars[c],
                    choices=multiareg_strategies)


            if c in ["A","B"]:
                self.params[f"{c}-preload"] = stage_param(
                        value=0, 
                        description=f"How many registers for {c} data to preload before the main loop",
                        default=0)

    def progress(self) -> list[composition_stage]:
        components = self.context.ukr_components

        addr_indices = { c :[i for i in range(int(self.params[f"{c}-addr-regs"].value))]
                        for c in components }

        addr_reg_counts = { c : int(self.params[f"{c}-addr-regs"].value)
                        for c in components }

        data_reg_counts = { c : int(self.params[f"{c}-data-regs"].value)
                           for c in components}

        if "gemm" == self.context.params["ukr"].value:
            addr_indices["alpha"] = [0]
            addr_indices["beta"] = [0]
            data_reg_counts["alpha"] = 1
            data_reg_counts["beta"] = 1
            addr_reg_counts["alpha"] = 1
            addr_reg_counts["beta"] = 1

        m = self.context.params["m"].value
        n = self.context.params["n"].value
        k = self.context.params["k"].value

        strides = {c : (None,None) for c in components}
        i = 0
        for char in components:
            comp_slist = [None,None]
            if self.params["row-strides"].value:
                if char in self.params["row-strides"].value:
                    stridelog.debug(f"Component {char} has row stride")
                    comp_slist[0] = i
                    i+= 1
            if self.params["column-strides"].value:
                if char in self.params["column-strides"].value:
                    stridelog.debug(f"Component {char} has col stride")
                    comp_slist[1] = i
                    i+= 1

            strides[char] = tuple(comp_slist)

        self.context.mappers['A'] = strided_mapper((m,1), strides['A'],vecdim=0)
        self.context.mappers['B'] = strided_mapper((1,n), strides['B'],vecdim=1)
        self.context.mappers['C'] = strided_mapper((m,n), strides['C'],vecdim=0)

        if "gemm" == self.context.params["ukr"].value:
            self.context.mappers['AB'] = strided_mapper((m,n), strides['AB'],vecdim=0)

        self.context.strides = strides

        scalar_mapper = same_address_mapper()

        self.context.mappers['beta'] = scalar_mapper
        self.context.mappers['alpha'] = scalar_mapper

        sup = self.context.sup

        component_tiles = {
            "A" : deepcopy(sup.a_tile),
            "B" : deepcopy(sup.b_tile),
            "C" : deepcopy(sup.c_tile)
        }
        if "gemm" == self.context.params["ukr"].value:
            component_tiles = component_tiles | {
            "AB" : deepcopy(sup.c_tile),
            "beta" : deepcopy(scalar_tile),
            "alpha" : deepcopy(scalar_tile)
        }

        real_tiles = deepcopy(component_tiles)
        if self.context.params["op"].value == 'fma' and "unvec-method" in self.context.params:
            real_from = "A"
            real_to = "B"
            if self.context.params["vecdir"].value == "N":
                real_from,real_to = real_to,real_from
            if self.context.params["unvec-method"].value in ['load_bcast']:
                real_tiles[real_to] = real_tiles[real_from]

        strats = { c : self.params[f"{c}-multiaddr-strat"].value \
                for c in components}

        strats['beta'] = "interleave"
        strats['alpha'] = "interleave"


        off_starts,off_ranges,off_steps = calculate_addr_parameters(
                sup=self.context.sup,
                primary_op=self.context.params["op"].value,
                gen=self.context.gen,
                addr_reg_counts=addr_reg_counts,
                data_tiles=component_tiles, 
                real_tiles=real_tiles, 
                narrow_components=["A","B"], 
                wide_components=["AB","C","beta","alpha"],
                m=m,n=n,k=k,
                vecdims={
                    "A" : 0,
                    "B" : 1,
                    "C" : 0,
                    "AB" : 0,
                    "beta" : 0,
                    "alpha" : 0,
                    },
                strides=strides,
                mappers=self.context.mappers,
                strats=strats)


        ar = addr_resolver(indices          = addr_indices,
                           starting_offsets = off_starts,
                           offset_ranges    = off_ranges,
                           steps            = off_steps,
                           max_incs=4)


        #TODO: investigate if there are architectures where this is relevant
        res_steps = {c:1 for c in components}

        preload_counts = {c : self.params[f"{c}-preload"].value for c in ["A","B"]}
        preload_counts["C"] = self.params[f"C-data-regs"].value
        if "gemm" == self.context.params["ukr"].value:
            preload_counts["AB"] = self.params[f"C-data-regs"].value
            preload_counts["beta"] = 0
            preload_counts["alpha"] = 0
            res_steps["alpha"] = 1
            res_steps["beta"] = 1


        resolve_order = deepcopy(components)
        if "gemm" == self.context.params["ukr"].value:
            resolve_order.extend(["beta","alpha"])

        self.context.model = load_store_cpu(
                res_counts=data_reg_counts,
                res_steps=res_steps,
                ar=ar,
                preload_counts=preload_counts,
                offset_mappers=self.context.mappers,
                #TODO: parameterize resolve order
                resolve_order=resolve_order)


        self.context.specializer.set_model(model=self.context.model)

        #print("\n".join(map(str,inspector(mm_ops_p1k))))


        mm_ops = self.context.tifs["mm"]
        mm_ops_p1k = self.context.tifs["mm_p1k"]
        mm_ops_p2k = self.context.tifs["mm_p2k"]

        self.context.irs["preload"] = self.context.model.preload(
                ops=mm_ops,next_ops=mm_ops_p1k,
                zero_components=["C","AB"],
                ignore_components=[])
        self.context.irs["main"] = self.context.model(mm_ops)
        self.context.irs["preload_next"] = self.context.model.preload(
                mm_ops_p1k,
                mm_ops_p2k,
                zero_addrs=False,
                zero_components=[],
                ignore_components=["C","AB"])

        if "gemm" == self.context.params["ukr"].value:
            betascale_ops = self.context.tifs["betascale"]
            alphascale_ops = self.context.tifs["alphascale"]
            self.context.irs["betascale"] = self.context.model(betascale_ops)
            self.context.irs["alphascale"] = self.context.model(alphascale_ops)

        self.context.irs["store"] = self.context.model.store_modified(ignore_components="AB")

        self.context.params.update(self.params)

        if "unvec-method" in self.context.params:
            return [unvec_lsc_stage]

        return list()
