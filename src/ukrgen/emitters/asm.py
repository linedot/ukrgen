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
from ..models.lsc.index import lsc_reg_index
from ..models.load_store_cpu import load_store_cpu

import re
from functools import singledispatchmethod
from copy import deepcopy


class lsc_asm_emitter(lsc_emitter):
    def __init__(self,
                 gen : asmgen,
                 rt : reg_tracker,
                 model : load_store_cpu,
                 data_registers    : dict[str,set[lsc_reg_index]],
                 address_registers : dict[str,set[lsc_reg_index]],
                 component_triples : list[tuple[str,str,str]],
                 data_tags         : dict[str,str],
                 offset_registry   : dict[str,set[lsc_offset]],
                 vindex_registry   : dict[str,set[lsc_offset]]):
        self.gen = gen
        self.rt = rt
        self.model = model
        self.address_registers = address_registers
        self.data_registers    = data_registers
        self.component_triples = component_triples
        self.data_tags         = data_tags
        self.offset_registry   = offset_registry
        self.vindex_registry   = vindex_registry

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
        try:
            residx = self.rt.aliased_regs[dreg_tag][f"RES:{op.res_idx}"]
        except KeyError as e:
            print(f"RES:{op.res_idx} not found in aliased regs for {dreg_tag}")
            print(f"Aliased {dreg_tag}s: ")
            for k,v in self.rt.aliased_regs[dreg_tag].items():
                print(f"{k} = {v}")
            
            print(f"data register set for {cr}:")
            for v in self.data_registers[cr]:
                print(f"{v}")
            raise e

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


    def calculate_offset(self, component : str,
                         off : lsc_offset,
                         component_dts : dict[str,adt],
                         ways : int) -> str:
        asmblock = ""


        dt_size = adt_size(component_dts[component])
        dt_shift = dt_size.bit_length()-1

        ways_shift = ways.bit_length()-1

        offreg_idx = self.rt.aliased_regs['greg'][f"{component}off:{str(off)}"]
        offreg = self.gen.greg(offreg_idx)
        asmblock += self.gen.zero_greg(greg=offreg)
        for sxv,value in off.sxv_strides.items():
            if 0 == value:
                continue
            tmpreg_idx = self.rt.reserve_any_reg('greg')
            tmpreg = self.gen.greg(tmpreg_idx)
            asmblock += self.gen.mov_greg_imm(reg=tmpreg, imm=value)

            for stride_id in sxv.stride_ids:
                alias = f"stride{stride_id}"
                if not alias in self.rt.aliased_regs['greg']:
                    idx = self.rt.reserve_any_reg('greg')
                    self.rt.alias_reg('greg', alias, idx)
                else:
                    idx = self.rt.aliased_regs['greg'][alias]

                reg = self.gen.greg(idx)
                asmblock += self.gen.mul_greg_greg(dst=tmpreg,reg1=tmpreg,reg2=reg)
            
            for vlen_id in sxv.vlen_ids:
                if vlen_id != 0:
                    raise NotImplementedError("Only first VLEN implemented right now")

                vlenidx = self.rt.aliased_regs['greg']['vlen']
                vlenreg = self.gen.greg(vlenidx)
                asmblock += self.gen.mul_greg_greg(dst=tmpreg,reg1=tmpreg,reg2=vlenreg)


            # ways would reduce the number of elements
            asmblock += self.gen.shift_greg_left(reg=tmpreg,bit_count=dt_shift-ways_shift)
            asmblock += self.gen.add_greg_greg(dst=offreg,reg1=offreg,reg2=tmpreg)
            self.rt.unuse_reg('greg', tmpreg_idx)

        for i,value in enumerate(off.reg_strides):
            if 0 == value:
                continue
            tmpreg_idx = self.rt.reserve_any_reg('greg')
            tmpreg = self.gen.greg(tmpreg_idx)
            asmblock += self.gen.mov_greg_imm(reg=tmpreg,imm=value)

            alias = f"stride{i}"
            if not alias in self.rt.aliased_regs['greg']:
                idx = self.rt.reserve_any_reg('greg')
                self.rt.alias_reg('greg', alias, idx)
            else:
                idx = self.rt.aliased_regs['greg'][alias]

            reg = self.gen.greg(idx)
            asmblock += self.gen.mul_greg_greg(dst=tmpreg,reg1=tmpreg,reg2=reg)

            asmblock += self.gen.shift_greg_left(reg=tmpreg,bit_count=dt_shift-ways_shift)            
            asmblock += self.gen.add_greg_greg(dst=offreg,reg1=offreg,reg2=tmpreg)
            self.rt.unuse_reg('greg', tmpreg_idx)

        for i,value in enumerate(off.vlen_strides):
            if 0 == value:
                continue
            if i != 0:
                raise NotImplementedError("Only first VLEN implemented right now")
            tmpreg_idx = self.rt.reserve_any_reg('greg')
            tmpreg = self.gen.greg(tmpreg_idx)
            asmblock += self.gen.mov_greg_imm(reg=tmpreg, imm=value)

            idx = self.rt.aliased_regs['greg']['vlen']

            reg = self.gen.greg(idx)
            asmblock += self.gen.mul_greg_greg(dst=tmpreg,reg1=tmpreg,reg2=reg)
            
            asmblock += self.gen.shift_greg_left(reg=tmpreg,bit_count=dt_shift-ways_shift)
            asmblock += self.gen.add_greg_greg(dst=offreg,reg1=offreg,reg2=tmpreg)
            self.rt.unuse_reg('greg', tmpreg_idx)

        if 0 != off.immoff:
            asmblock += self.gen.add_greg_imm(reg=offreg,imm=off.immoff*dt_size)
        return asmblock

    def init_vlenregs(self, component_dts : dict[str,adt]) -> str:

        asmblock = ""

        # determine narrowest type:
        narrow_dt = list(component_dts.values())[0]
        min_size = adt_size(narrow_dt)

        for a,b,c in self.component_triples:
            size = adt_size(component_dts[a])
            if size < min_size:
                min_size = size
                narrow_dt = component_dts[a]

        if 'vlen' not in self.rt.aliased_regs['greg']:
            vlenidx = self.rt.reserve_any_reg('greg')
            self.rt.alias_reg('greg', 'vlen', vlenidx)
            vlenreg = self.gen.greg(vlenidx)
            asmblock += self.gen.simd_size_to_greg(reg=vlenreg, dt=narrow_dt)
        else:
            vlenidx = self.rt.aliased_regs['greg']['vlen']
            vlenreg = self.gen.greg(vlenidx)

        for component in self.offset_registry.keys():
            for off in self.offset_registry[component]:
                if not off.is_vector:
                    continue

                alias = f"{component}off:{str(off)}"

                dt_shift = adt_size(component_dts[component]).bit_length()-1

                vlenxnidx = self.rt.reserve_any_reg('greg')
                self.rt.alias_reg('greg', alias, vlenxnidx)

                vlenxnreg = self.gen.greg(vlenxnidx)

                asmblock += self.gen.mov_greg(src=vlenreg,dst=vlenxnreg)
                asmblock += self.gen.shift_greg_left(reg=vlenxnreg, bit_count=dt_shift)

                vlenxn = off.vlen_strides[0]
                if vlenxn > 1:
                    asmblock += self.gen.mul_greg_imm(src=vlenreg,
                                                      dst=vlenxnreg,
                                                      factor=vlenxn)

        return asmblock

    #TODO: consider deduplication. This is only needed for intialization offset
    #      in code_init(), might be handled by specializer instead?
    def register_offset(self, component : str, off : lsc_offset):
        if off == lsc_offset.zero_offset():
            # TODO: investigate why this can happen
            return
        #self.stridelog.debug(f"registering offset {off} for component={component}")
        if component not in self.offset_registry:
            self.offset_registry[component] = set()
        if off in self.offset_registry[component]:
            return
        self.offset_registry[component].add(off)

    def code_init(self, component_dts : dict[str,adt]) -> str:

        #TODO: datatype for quirks. For now go with the narrow type
        
        # determine narrowest type:
        narrow_dt = list(component_dts.values())[0]
        min_size = adt_size(narrow_dt)

        for a,b,c in self.component_triples:
            size = adt_size(component_dts[a])
            if size < min_size:
                min_size = size
                narrow_dt = component_dts[a]
                        
        asmblock = self.gen.isaquirks(rt=self.rt,dt=narrow_dt)

        asmblock += self.init_vlenregs(component_dts=component_dts)


        
        acc_components = set()
        for a,b,c in self.component_triples:
            acc_components.add(c)


        # Register offsets for calculating starting addresses:
        for component in self.address_registers.keys():
            for aidx in self.address_registers[component]:
                if aidx.indices[0] == 0:
                    continue
                sos = self.model.ar.starting_offsets[component]
                step = sos[aidx.indices[0]] - sos[aidx.indices[0]-1]
                self.register_offset(component=component, off=step)
                #print(f"calculated {step} for {component}")


        # reserve offset registers
        for component in self.offset_registry.keys():
            for off in self.offset_registry[component]:
                if off.is_scalar:
                    continue
                
                if off.is_vector:
                    factor = off.vlen_strides[0]
                    if factor < self.gen.max_add_voff and factor > 0:
                        continue

                # determine largest widening ways:
                ways = 1
                for a,b,c in self.component_triples:
                    narrow_size = adt_size(component_dts[a])
                    if c == component:
                        ways = max(adt_size(component_dts[c])//narrow_size,ways)

                regidx = self.rt.reserve_any_reg('greg')
                alias = f"{component}off:{str(off)}"
                self.rt.alias_reg('greg', alias , regidx)
                #print(f"reserving GP reg {regidx} for {alias}")
                asmblock += self.gen.asmwrap(f"# {self.gen.greg(regidx)} = {alias}")
                asmblock += self.gen.asmwrap(f"# calculation -->")
                asmblock += self.calculate_offset(component, off,
                                                  component_dts=component_dts,
                                                  ways=ways)
                asmblock += self.gen.asmwrap(f"# calculation end <--")


        # reserve vector indices
        for component in self.vindex_registry.keys():
            for stride in self.vindex_registry[component]:
                stvidx = self.rt.reserve_any_reg("vreg")
                alias = f"{component}vidx:{str(stride)}"
                self.rt.alias_reg('vreg', alias, stvidx)
                #print(f"reserving vreg {stvidx} for {alias}")

                galias = f"{component}off:{str(stride)}"
                stridx = self.rt.aliased_regs["greg"][galias]
                streg = self.gen.greg(stridx)
                stvreg = self.gen.vreg(stvidx)
                asmblock += self.gen.asmwrap(f"# {self.gen.vreg(stvidx)} = {alias}")
                asmblock += self.gen.asmwrap(f"# calculation -->")
                asmblock += self.gen.greg_to_voffs(streg=streg, vreg=stvreg,
                                                   dt=component_dts[component])
                asmblock += self.gen.asmwrap(f"# calculation end <--")

        # reserve address registers
        for component in self.address_registers.keys():
            for aidx in self.address_registers[component]:
                aregblock = ""

                sos = self.model.ar.starting_offsets[component]
                step = sos[aidx.indices[0]] - sos[aidx.indices[0]-1]

                reg_alias = f"ADDR:{aidx}"
                #print(f"processing {reg_alias}")
                if reg_alias not in self.rt.aliased_regs['greg']:
                    aregidx = self.rt.reserve_any_reg('greg')
                    self.rt.alias_reg('greg', reg_alias, aregidx)
                    
                    # Compute starting address

                    # Not deeopcopying this will mess up existing ops
                    prev_idx = deepcopy(aidx)
                    prev_idx.indices[0] -=1
                    prev_alias = f"ADDR:{prev_idx}"

                    prev_reg_idx = self.rt.aliased_regs['greg'][prev_alias]

                    aregblock += self.gen.mov_greg(
                        src=self.gen.greg(prev_reg_idx),
                        dst=self.gen.greg(aregidx)
                    )
                    model_state = self.model.states[self.model.current_state]
                    t = model_state.last_tile_used[component]

                    if step != lsc_offset.zero_offset():
                        aregblock += self.transform(
                            lsc_addr_add(
                                component=component,
                                addr_idx=aidx.indices[0],
                                off=step,
                                t=t),
                            component_dts=component_dts)

                else:
                    aregidx = self.rt.aliased_regs['greg'][reg_alias]
                #print(f"reserving GP reg {aregidx} for {reg_alias}")

                # asmblock += f"// todo: init {reg_alias}\n"
                asmblock += self.gen.asmwrap(f"# {self.gen.greg(aregidx)} = {reg_alias}")
                asmblock += self.gen.asmwrap(f"# calculation -->")
                asmblock += aregblock
                asmblock += self.gen.asmwrap(f"# calculation end <--")


        # reserve data registers
        for component in self.data_registers.keys():
            for didx in self.data_registers[component]:
                reg_alias = f"RES:{lsc_reg_index(component,[didx])}"

                # inaccurate, unvec not considered
                #model_state = self.model.states[self.model.current_state]
                #t = model_state.last_tile_used[component]
                #dreg_tag = determine_dreg_tag(t.dima, t.dimb)
                dreg_tag = self.data_tags[component]

                dregidx = self.rt.reserve_any_reg(dreg_tag)

                self.rt.alias_reg(dreg_tag, reg_alias, dregidx)
                #print(f"reserving {dreg_tag} {dregidx} for {reg_alias}")

                dt = component_dts[component]

                if dreg_tag in ["freg","treg"]:
                    dregname = str(getattr(self.gen,dreg_tag)(dregidx,dt))
                else:
                    dregname = str(getattr(self.gen,dreg_tag)(dregidx))

                # asmblock += f"// todo: init {reg_alias}\n"
                asmblock += self.gen.asmwrap(f"# {dregname} = {reg_alias}")

        return asmblock

    def code_fini(self, component_dts : dict[str,adt]) -> str:
        # determine narrowest type:
        narrow_dt = list(component_dts.values())[0]
        min_size = adt_size(narrow_dt)

        for a,b,c in self.component_triples:
            size = adt_size(component_dts[a])
            if size < min_size:
                min_size = size
                narrow_dt = component_dts[a]

        return self.gen.isaendquirks(rt=self.rt,dt=narrow_dt)
