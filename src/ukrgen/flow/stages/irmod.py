# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import sys
import logging
from copy import deepcopy

from .stage import stage
from ..ukr_context import ukr_context
from ..stage_param import stage_param

from ...specializers.asm import op_support
from ...models.lsc.offset import lsc_offset
from ...models.load_store_operations import (
        lsc_load,
        lsc_store,
        lsc_addr_add,
        lsc_transformation,
        ldst_modifier
    )

# TODO: this whole stage is basically NEON-only, investigate if it is useful for
#       any other ISA or architecture. (Possibly also useful for ISAs that don't
#       have a combined ldst+addr add but have micro-op-fusion of some kind. In
#       that case it could make sense to reschedule the ops. Might also be done 
#       by a pattern-prefering scheduler)
class ldinc_lsc_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)

        self.debug = logging.getLogger("LSCIRMOD").debug

    def progress(self) -> list[stage]:

        zo = lsc_offset.zero_offset()
        sup = self.context.sup

        for key,ir in self.context.irs.items():
            index_pairs = []
            for i,ldst in enumerate(ir):
                if not isinstance(ldst, (lsc_load,lsc_store)):
                    continue
                # Only combine if the offset is not zero
                if zo != ldst.off:
                    continue
                for j,add in enumerate(ir[i+1:],start=i+1):
                    # encountered another ldst before encountering an add,
                    # can't combine, therefore abort
                    # TODO: Check for other cases where we need to abort,
                    #       maybe if the addr idx is in reads but is not 
                    #       an addr add?
                    if isinstance(add, (lsc_load,lsc_store)):
                        if add.addr_idx == ldst.addr_idx:
                            break
                    # this isn't an add, continue
                    if not isinstance(add, lsc_addr_add):
                        continue
                    if ldst.addr_idx != add.addr_idx:
                        continue

                    data_not_scalar = ldst.mods.isdisjoint(
                            {ldst_modifier.lane, ldst_modifier.bcast1})

                    # Actually this would be an ldr, so it's fine
                    # data_not_scalar = data_not_scalar and not op.t.is_scalar

                    # IIUC in neon the immediate post-increment MUST BE
                    # the size of the data for ld1r/ld1/st1,
                    # but can be [-256,255] for ldr/str
                    if add.off.is_regstride:
                        if data_not_scalar:
                            # No register post-index ldr/str
                            break
                        index_pairs.append((i,j))
                    elif add.off.is_scalar:
                        if (add.off.immoff == 1) or data_not_scalar:
                            index_pairs.append((i,j))
                    # If we can't combine with the first addr add we found
                    # we have to give up combining this ld/st as the address has
                    # been modified from this op on
                    break

            for ldst_idx,add_idx in index_pairs:
                self.debug(f"combining {ldst_idx}:{ir[ldst_idx]} with {add_idx}:{ir[add_idx]}")
                new_op = deepcopy(ir[ldst_idx])
                new_op.off = ir[add_idx].off
                new_op.mods = ir[ldst_idx].mods.union(
                        {ldst_modifier.postinc})
                # Add addr idx to writes
                new_op.writes.append(0)
                new_op.writes = sorted(new_op.writes)
                ir[ldst_idx] = new_op
                ir[add_idx] = None

            # Filter out deleted adds
            self.context.irs[key] = [op for op in ir if op is not None]
        
        self.context.params.update(self.params)

        self.debug("################### LDSTINC PSEUDO-ASM ###################")
        self.debug("\n".join(map(str,self.context.irs["preload"])))
        self.debug("MAIN LOOP -------------------------------")
        self.debug("  "+"\n  ".join(map(str,self.context.irs["main"])))
        self.debug("PRELOAD NEXT ----------------------------")
        self.debug("  "+"\n  ".join(map(str,self.context.irs["preload_next"])))
        self.debug("END MAIN LOOP ---------------------------")
        self.debug("LASTITER --------------------------------")
        self.debug("\n".join(map(str,self.context.irs["lastiter"])))
        self.debug("END LASTITER ----------------------------")
        k = int(self.context.params["k"].value)
        if k > 1:
            self.debug("K1 PRELOAD ------------------------------")
            self.debug("  "+"\n  ".join(map(str,self.context.irs["1k_preload"])))
            self.debug("K1 LOOP ---------------------------------")
            self.debug("K1 MAIN ---------------------------------")
            self.debug("  "+"\n  ".join(map(str,self.context.irs["1k_main"])))
            self.debug("K1 PRELOAD_NEXT -------------------------")
            self.debug("  "+"\n  ".join(map(str,self.context.irs["1k_preload_next"])))
            self.debug("END K1 LOOP -----------------------------")
            self.debug("K1 LASTITER -----------------------------")
            self.debug("\n".join(map(str,self.context.irs["1k_lastiter"])))
            self.debug("END K1 LOOP -----------------------------")
        self.debug("STOREBLOCK ------------------------------")
        self.debug("\n".join(map(str,self.context.irs["store"])))
        self.debug("ENDSTOREBLOCK ---------------------------")

        return list()



class unvec_lsc_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)

        self.debug = logging.getLogger("LSC").debug

    def progress(self) -> list[stage]:

        sup = self.context.sup
        vecdir = self.context.params["vecdir"].value

        scalar_input = "B"
        if vecdir == "M":
            scalar_input = "B"
        elif vecdir == "N":
            scalar_input = "A"
        unvec_components = [scalar_input,"alpha","beta"]

        if "load_bcast" == self.context.params["unvec-method"].value:
            vec_tile = sup.c_tile
            def mod_load(op : lsc_load) -> lsc_load:
                if not isinstance(op,lsc_load):
                    return op
                if op.addr_idx.component in unvec_components:
                    op.mods.add(ldst_modifier.bcast1)
                    op.tiles[1] = vec_tile

                return op
            def mod_transform(op : lsc_transformation) -> lsc_transformation:
                if not isinstance(op,lsc_transformation):
                    return op
                for i,idx in enumerate(op.indices):
                    c = idx.component
                    if c in unvec_components:
                        op.tiles[i] = vec_tile

                return op

            for mod in [mod_load,mod_transform]:
                for key,ir in self.context.irs.items():
                    self.context.irs[key] = [mod(op) for op in self.context.irs[key]]
        
        self.context.params.update(self.params)

        self.debug("################### UNVEC PSEUDO-ASM ###################")
        self.debug("\n".join(map(str,self.context.irs["preload"])))
        self.debug("MAIN LOOP -------------------------------")
        self.debug("  "+"\n  ".join(map(str,self.context.irs["main"])))
        self.debug("PRELOAD NEXT ----------------------------")
        self.debug("  "+"\n  ".join(map(str,self.context.irs["preload_next"])))
        self.debug("END MAIN LOOP ---------------------------")
        self.debug("LASTITER --------------------------------")
        self.debug("\n".join(map(str,self.context.irs["lastiter"])))
        self.debug("END LASTITER ----------------------------")
        k = int(self.context.params["k"].value)
        if k > 1:
            self.debug("K1 PRELOAD ------------------------------")
            self.debug("  "+"\n  ".join(map(str,self.context.irs["1k_preload"])))
            self.debug("K1 LOOP ---------------------------------")
            self.debug("K1 MAIN ---------------------------------")
            self.debug("  "+"\n  ".join(map(str,self.context.irs["1k_main"])))
            self.debug("K1 PRELOAD_NEXT -------------------------")
            self.debug("  "+"\n  ".join(map(str,self.context.irs["1k_preload_next"])))
            self.debug("END K1 LOOP -----------------------------")
            self.debug("K1 LASTITER -----------------------------")
            self.debug("\n".join(map(str,self.context.irs["1k_lastiter"])))
            self.debug("END K1 LOOP -----------------------------")
        self.debug("STOREBLOCK ------------------------------")
        self.debug("\n".join(map(str,self.context.irs["store"])))
        self.debug("ENDSTOREBLOCK ---------------------------")

        return list()

class irmod_inserter_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)

        self.debug = logging.getLogger("LSCIRMOD").debug

    def progress(self) -> list[stage]:
        
        self.context.params.update(self.params)

        #TODO: detection/parameterization instead of hardcoding

        if "neon" == self.context.params["isa"].value:
            return [ldinc_lsc_stage]

        return list()
