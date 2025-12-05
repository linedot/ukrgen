# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import sys
import logging

from .composition import composition_stage
from ..gemm import gemm_context
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
class ldinc_lsc_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        self.debug = logging.getLogger("LSCIRMOD").debug

    def progress(self) -> list[composition_stage]:

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
                        index_pairs.append((i,j))
                    elif add.off.is_scalar:
                        if (add.off.immoff == 1) or data_not_scalar:
                            index_pairs.append((i,j))
                    # If we can't combine with the first addr add we found
                    # we have to give up combining this ld/st as the address has
                    # been modified from this op on
                    break

            for ldst_idx,add_idx in index_pairs:
                self.debug(f"combining {ir[ldst_idx]} with {ir[add_idx]}")
                ir[ldst_idx].off = ir[add_idx].off
                ir[ldst_idx].mods = ir[ldst_idx].mods.union(
                        {ldst_modifier.postinc})
                ir[add_idx] = None

            # Filter out deleted adds
            self.context.irs[key] = [op for op in ir if op is not None]
        
        self.context.params.update(self.params)

        self.debug("################### PSEUDO-ASM ###################")
        self.debug("\n".join(map(str,self.context.irs["preload"])))
        self.debug("MAIN LOOP -------------------------------")
        self.debug("  "+"\n  ".join(map(str,self.context.irs["main"])))
        self.debug("PRELOAD NEXT ----------------------------")
        self.debug("  "+"\n  ".join(map(str,self.context.irs["preload_next"])))
        self.debug("END MAIN LOOP ---------------------------")
        if "gemm" == self.context.params["ukr"].value:
            self.debug("BETASCALE BLOCK -------------------------")
            self.debug("\n".join(map(str,self.context.irs["betascale"])))
            self.debug("END BETASCALE BLOCK ---------------------")
            self.debug("ALPHASCALE BLOCK ------------------------")
            self.debug("\n".join(map(str,self.context.irs["alphascale"])))
            self.debug("END ALPHASCALE BLOCK --------------------")
        self.debug("STOREBLOCK ------------------------------")
        self.debug("\n".join(map(str,self.context.irs["store"])))
        self.debug("ENDSTOREBLOCK ---------------------------")

        return list()



class unvec_lsc_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        self.debug = logging.getLogger("LSC").debug

    def progress(self) -> list[composition_stage]:

        sup = self.context.sup
        unvec_components = ["B","alpha","beta"]
        if "load_bcast" == self.context.params["unvec-method"].value:
            vec_tile = sup.a_tile
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

        self.debug("################### PSEUDO-ASM ###################")
        self.debug("\n".join(map(str,self.context.irs["preload"])))
        self.debug("MAIN LOOP -------------------------------")
        self.debug("  "+"\n  ".join(map(str,self.context.irs["main"])))
        self.debug("PRELOAD NEXT ----------------------------")
        self.debug("  "+"\n  ".join(map(str,self.context.irs["preload_next"])))
        self.debug("END MAIN LOOP ---------------------------")
        if "gemm" == self.context.params["ukr"].value:
            self.debug("BETASCALE BLOCK -------------------------")
            self.debug("\n".join(map(str,self.context.irs["betascale"])))
            self.debug("END BETASCALE BLOCK ---------------------")
            self.debug("ALPHASCALE BLOCK ------------------------")
            self.debug("\n".join(map(str,self.context.irs["alphascale"])))
            self.debug("END ALPHASCALE BLOCK --------------------")
        self.debug("STOREBLOCK ------------------------------")
        self.debug("\n".join(map(str,self.context.irs["store"])))
        self.debug("ENDSTOREBLOCK ---------------------------")

        return list()

class irmod_inserter_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        self.debug = logging.getLogger("LSCIRMOD").debug

    def progress(self) -> list[composition_stage]:
        
        self.context.params.update(self.params)

        #TODO: detection/parameterization instead of hardcoding

        if "neon" == self.context.params["isa"].value:
            return [ldinc_lsc_stage]

        return list()
