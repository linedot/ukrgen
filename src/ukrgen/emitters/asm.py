from ..models.load_store_operations import (
        lsc_add_val_off,
        lsc_addr_add,
        lsc_debugmsg,
        lsc_load,
        lsc_store,
        lsc_transformation,
        lsc_zero,
        ldst_modifier
        )

from ..models.loop import lsc_loop

from ..models.lsc.offset import lsc_offset

from ..specializers.special_treg_ldst import (
        lsc_treg_row_load,
        lsc_treg_row_store
        )

from ..components.tile import determine_dreg_tag,dimension_type

from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import asm_data_type as adt,reg_tracker,adt_size
from asmgen.asmblocks.operations import modifier as opmod

from .lsc import lsc_emitter

import re
from functools import singledispatchmethod


class lsc_asm_emitter(lsc_emitter):
    def __init__(self,
                 gen : asmgen,
                 rt : reg_tracker):
        self.gen = gen
        self.rt = rt

    @singledispatchmethod
    def transform(self, op, component_dts : dict[str,adt]) -> str:
        pass

    @transform.register
    def _(self, op : lsc_add_val_off, component_dts : dict[str,adt]) -> str:
        if not op.off.is_scalar:
            raise NotImplementedError(
                "Non-scalar offsets not implemented for non-address values")
        alias = op.valname
        regidx = self.rt.aliased_regs['greg'][alias]
        reg = self.gen.greg(regidx)
        return self.gen.add_greg_imm(
            reg=reg,
            imm=op.off.immoff)

    @transform.register
    def _(self, op : lsc_addr_add, component_dts : dict[str,adt]) -> str:
        alias = f"ADDR:{op.addr_idx}"

        aregidx = self.rt.aliased_regs['greg'][alias]
        areg = self.gen.greg(aregidx)
        data_t = op.tiles[1]
        #tsize=data_t.dima.size*data_t.dimb.size
        ca = op.indices[0].component
        dt = component_dts[ca]
        dt_bytes = adt_size(dt)

        if op.off == lsc_offset.zero_offset():
            return ""

        if op.off.is_vector:
            #factor = op.off.vlen_strides[0]*tsize
            factor = op.off.vlen_strides[0]

            if factor < self.gen.max_add_voff and factor > 0:
                return self.gen.add_greg_voff(reg=areg, 
                                              offset=factor, 
                                              dt=dt)

            vxnalias = f"{ca}off:{str(op.off)}"
            vlen_idx = self.rt.aliased_regs['greg'][vxnalias]
            vlenreg = self.gen.greg(vlen_idx)
            return self.gen.add_greg_greg(dst=areg,
                                          reg1=areg,
                                          reg2=vlenreg)

        elif op.off.is_scalar:
            return self.gen.add_greg_imm(
                reg=areg,
                #imm=op.off.immoff*tsize*dt_bytes)
                imm=op.off.immoff*dt_bytes)

        else:
            regidx = self.rt.aliased_regs['greg'][f"{ca}off:{str(op.off)}"]
            offreg = self.gen.greg(regidx)
            return self.gen.add_greg_greg(dst=areg,
                                          reg1=areg,
                                          reg2=offreg)


    def transform_ldst(self, op : Union[lsc_load,lsc_store],
                       component_dts : dict[str,adt],
                       action : str):

        aregidx = self.rt.aliased_regs['greg'][f"ADDR:{op.addr_idx}"]
        areg = self.gen.greg(aregidx)

        ca = op.indices[0].component
        dt = component_dts[ca]
        dt_bytes = adt_size(dt)



        dreg_tag = determine_dreg_tag(op.t.dima, op.t.dimb)

        vlenmul = 0;
        if op.t.dima.dt == dimension_type.vla:
            vlenmul += 1
        if op.t.dimb.dt == dimension_type.vla:
            vlenmul += 1
            
        # register could have been replaced
        # TODO: cleaner method (perhaps op.addr_component/ op.res_component ?)
        #residx = self.rt.aliased_regs[dreg_tag][f"{op.component}{op.res_idx}"]
        cr = op.indices[1].component
        alias = f"RES:{op.res_idx}"
        #print(f"searching index for {alias} in {dreg_tag} aliases:")
        residx = self.rt.aliased_regs[dreg_tag][f"RES:{op.res_idx}"]
        #print(f"found: {residx}")

        if dreg_tag in ["freg","treg"]:
            dreg = getattr(self.gen,dreg_tag)(residx,dt)
        else:
            dreg = getattr(self.gen,dreg_tag)(residx)

        
        #TODO:
        # - differentiate voffset,immoffset, nonoffset, ...
        # - differentiate tile/vreg/freg

        kwargs = dict()
        suffix = ""
        if ldst_modifier.bcast1 in op.mods:
            suffix += "_bcast1"
        if ldst_modifier.lane in op.mods:
            suffix += "_lane"
            kwargs["lane"] = op.properties["lane"]
        if ldst_modifier.postinc in op.mods:
            if op.off.is_scalar:
                suffix += "_inc"
                kwargs["offset"] = op.off.immoff*dt_bytes
            elif op.off.is_regstride:
                suffix += "_greginc"
                offreg_idx = self.rt.aliased_regs['greg'][f"{ca}off:{str(op.off)}"]
                kwargs["offreg"] = self.gen.greg(offreg_idx)

        if op.stride is None or \
                (ldst_modifier.lane in op.mods) or \
                (dreg_tag == "freg"):
            
            if lsc_offset.zero_offset() == op.off or ldst_modifier.postinc in op.mods:
                if 'treg' == dreg_tag:
                    lsfunc = getattr(self.gen, f"{action}_tile{suffix}")
                    kwargs["areg"] = areg
                    kwargs["treg"] = dreg
                    kwargs["dt"] = dt
                    return lsfunc(**kwargs)
                if 'vreg' == dreg_tag:
                    lsfunc = getattr(self.gen, f"{action}_vector{suffix}")
                    kwargs["areg"] = areg
                    kwargs["vreg"] = dreg
                    kwargs["dt"] = dt
                    return lsfunc(**kwargs)

            if 'freg' == dreg_tag:
                lsfunc = getattr(self.gen, f"{action}_scalar{suffix}_immoff")

                return lsfunc(
                        areg=areg,
                        offset=op.off.immoff*dt_bytes,
                        freg=dreg,
                        dt=dt)

            if 'vreg' == dreg_tag:
                if vlenmul > 0 and ldst_modifier.bcast1 not in op.mods:
                    lsfunc = getattr(self.gen, f"{action}_vector{suffix}_voff")
                    return lsfunc(areg=areg, voffset=op.off.vlen_strides[0], vreg=dreg, dt=dt)
                else:
                    lsfunc = getattr(self.gen, f"{action}_vector{suffix}_immoff")
                    return lsfunc(areg=areg, offset=op.off.immoff*dt_bytes, vreg=dreg, dt=dt)

        else:
            if lsc_offset.zero_offset() == op.off:
                if 'vreg' == dreg_tag:
                    lsfunc = getattr(self.gen, f"{action}_vector{suffix}_gregstride")
                    
                    streg_idx = self.rt.aliased_regs['greg'][f"{ca}off:{str(op.stride)}"]
                    try:
                        return lsfunc(areg=areg, sreg=self.gen.greg(streg_idx), vreg=dreg, dt=dt)
                    except NotImplementedError as e:
                        gasc_suf = "gather"
                        if "store" == action:
                            gasc_suf = "scatter"
                        lsfunc = getattr(self.gen, f"{action}_vector{suffix}_{gasc_suf}")
                        stvidx = self.rt.aliased_regs['vreg'][f"{ca}vidx:{str(op.stride)}"]
                        it = it_from_dt_samesize(dt)
                        return lsfunc(areg=areg, offvreg=self.gen.vreg(stvidx), vreg=dreg, dt=dt, it=it)



        #raise NotImplementedError("Load not implemented yet")
        return self.gen.asmwrap(f"fixme: {action} {dreg_tag} with off {op.off} and stride {op.stride} not implemented")
    
        
    @transform.register
    def _(self, op : lsc_load, component_dts : dict[str,adt]) -> str:
        pass
        return self.transform_ldst(op=op,component_dts=component_dts,action="load")

    @transform.register
    def _(self, op : lsc_store, component_dts : dict[str,adt]) -> str:
        return self.transform_ldst(op=op,component_dts=component_dts,action="store")


    @transform.register
    def _(self, op : lsc_debugmsg, component_dts : dict[str,adt]) -> str:
        return self.gen.asmwrap("# "+op.msg)

    @transform.register
    def _(self, op : lsc_transformation, component_dts : dict[str,adt]) -> str:

        dregs = []

        dreg_tags = []

        a_dt = component_dts[op.data_components[0]]
        b_dt = component_dts[op.data_components[1]]
        c_dt = component_dts[op.data_components[2]]
        
        for i,(idx,t) in enumerate(zip(op.indices,
                                     op.tiles)):
            component = idx.component
            dcomponent = op.data_components[i]

            dreg_tag = determine_dreg_tag(t.dima, t.dimb)

            residx = self.rt.aliased_regs[dreg_tag][f"RES:{idx}"]

            if dreg_tag in ["freg","treg"]:
                dreg = getattr(self.gen,dreg_tag)(residx,component_dts[dcomponent])
            else:
                dreg = getattr(self.gen,dreg_tag)(residx)

            dregs.append(dreg)
            dreg_tags.append(dreg_tag)

            # TODO: subidx

        more_args = dict()
        modifiers = set()

        opstr = op.op
        # If there is a number at the end of the str, that's a partial instruction
        numbers = re.findall(r'\d+',opstr)
        if numbers:
            modifiers.add(opmod.PART)
            more_args['part'] = int(numbers[0])

            for number in numbers:
                opstr = opstr.replace(number, '')

        #if there is only 1 freg, this is a VF operation
        if 1 == dreg_tags.count('freg'):
            modifiers.add(opmod.VF)
            # TODO: separate hardware tiles (sup) from kernel tiles
            #       i.e. what changes with vecdir and handle this properly
            if dreg_tags[0] == 'freg':
                dregs[0],dregs[1] = dregs[1],dregs[0]

        #if tr_modifier.np in op.mods:
        #    modifiers.add(opmod.NP)

        arith_op = getattr(self.gen,opstr)

        return arith_op(adreg=dregs[0],bdreg=dregs[1],cdreg=dregs[2],
                        a_dt = a_dt, b_dt = b_dt, c_dt = c_dt,
                        modifiers=modifiers, **more_args)


    @transform.register
    def _(self, op : lsc_zero, component_dts : dict[str,adt]) -> str:

        

        dreg_tag = determine_dreg_tag(op.t.dima, op.t.dimb)

        cr = op.indices[0].component
        residx = self.rt.aliased_regs[dreg_tag][f"RES:{op.res_idx}"]

        dt = component_dts[cr]

        if dreg_tag == 'freg':
            dreg = self.gen.freg(residx, dt)
            return self.gen.zero_freg(freg=dreg, dt=dt)
        elif dreg_tag == 'vreg':
            dreg = self.gen.vreg(residx)
            return self.gen.zero_vreg(vreg=dreg, dt=dt)
        elif dreg_tag == 'treg':
            dreg = self.gen.treg(residx, dt)
            return self.gen.zero_treg(treg=dreg, dt=dt)

        raise NotImplementedError(f"Implementation for zeroing {dreg_tag} missing")
        pass

    @transform.register
    def _(self, op : lsc_treg_row_load, component_dts : dict[str,adt]) -> str:
        aregidx = self.rt.aliased_regs['greg'][f"{op.component}areg{op.addr_idx}"]
        areg = self.gen.greg(aregidx)

        dt = component_dts[op.component]

        asmblock = ""

        if 'rreg' not in self.rt.aliased_regs['greg']:
            # TODO: This needs to be 12-15 for SME. Figure out a way to 
            #       have generic register allocation restrictions
            # TODO: also potentially need multiple regs for different components?
            rreg_id = 12
            self.rt.reserve_specific_reg('greg',rreg_id)
            self.rt.alias_reg('greg', 'rreg', rreg_id)
            rreg = self.gen.greg(rreg_id)
            asmblock += self.gen.mov_greg_imm(reg=rreg, imm=0)

        rreg_id = self.rt.aliased_regs['rreg']
        rreg = self.gen.greg(rreg_id)

        residx = self.rt.aliased_regs['treg'][f"{op.component}{op.res_idx}"]

        asmblock += self.gen.load_tile_row(
                       areg=areg, 
                       rreg=rreg,
                       roff=op.roff,
                       voff=op.aoff,
                       treg=self.gen.treg(residx),
                       dt=dt)
        return asmblock

    @transform.register
    def _(self, op : lsc_treg_row_store, component_dts : dict[str,adt]) -> str:
        aregidx = self.rt.aliased_regs['greg'][f"{op.component}areg{op.addr_idx}"]
        areg = self.gen.greg(aregidx)

        dt = component_dts[op.component]

        asmblock = ""

        if 'rreg' not in self.rt.aliased_regs['greg']:
            # TODO: This needs to be 12-15 for SME. Figure out a way to 
            #       have generic register allocation restrictions
            rreg_id = 12
            self.rt.reserve_specific_reg('greg',rreg_id)
            self.rt.alias_reg('greg', 'rreg', rreg_id)
            rreg = self.gen.greg(rreg_id)
            asmblock += self.gen.mov_greg_imm(reg=rreg, imm=0)

        rreg_id = self.rt.aliased_regs['greg']['rreg']
        rreg = self.gen.greg(rreg_id)

        residx = self.rt.aliased_regs['treg'][f"{op.component}{op.res_idx}"]

        asmblock += self.gen.store_tile_row(
                       areg=areg, 
                       rreg=rreg,
                       roff=op.roff,
                       voff=op.aoff,
                       treg=self.gen.treg(residx, dt=dt),
                       dt=dt)
        return asmblock

    @transform.register
    def _(self, op : lsc_loop, component_dts : dict[str,adt]) -> str:

        indent = lambda i : (i+op.level)*"  "
        asmblock = ""
        for div in reversed(op.divergences):
            asmblock += indent(0) + self.gen.label(label = f"{op.name}_{div.name}")
            for subop in div.ops:
                asmblock += indent(1) + self.transform(subop,component_dts)

        
        cntr_idx = self.rt.aliased_regs["greg"][op.condition.first]
        asmblock += indent(0) + self.gen.label(label=op.name)

        for subop in op.block:
            asmblock += indent(1) + self.transform(subop,component_dts)

        for div in op.divergences:
            label = f"{op.name}_{div.name}"
            reg1 = self.gen.greg(self.rt.aliased_regs["greg"][div.condition.first])
            reg2 = None
            if div.condition.second is not None:
                reg2 = self.gen.greg(self.rt.aliased_regs["greg"][div.condition.second])
            asmblock += indent(1) + self.gen.cb(reg1 = reg1,
                                           reg2 = reg2,
                                           cmp=div.condition.comparison.cmp,
                                           label=label)


        reg1 = self.gen.greg(self.rt.aliased_regs["greg"][op.condition.first])
        reg2 = None
        if op.condition.second is not None:
            reg2 = self.gen.greg(self.rt.aliased_regs["greg"][op.condition.second])

        asmblock += indent(1) + self.gen.cb(reg1 = reg1,
                                            reg2 = reg2,
                                            cmp=op.condition.comparison.cmp,
                                            label=op.name)

        return asmblock
