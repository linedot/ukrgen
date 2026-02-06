# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from __future__ import annotations

from ..components.tile import determine_dreg_tag
from ..models.load_store_operations import (
        lsc_operation,
        lsc_load,
        lsc_store,
        lsc_zero,
        lsc_transformation,
        lsc_add_val_off,
        lsc_addr_add,
        )
from ..models.lsc.reg import lsc_reg_type

from .dependency import dependency_type,get_dependencies

from enum import Enum
from dataclasses import dataclass

@dataclass
class distance_specification:
    """
    Specifies the distance for the scheduler to maintain between instructions
    and the circumstances under which to maintain it

    :param dep_type: Type of dependency (read-after-read, read-after-write,
                     etc..) for which the distance is to be maintained. When
                     set to None, applies to all dependency types
    :type dep_type: class:`dependency_type|None`
    :param reg_type: type of register in the load-store cpu model
                     (data, address, value, ...). When set to None,
                     applies to all register types
    :type reg_type: class:`lsc_reg_type|None`
    :param asm_reg_tag: type of asm reg type as string, can be "greg", "freg",
                        "vreg" or "treg". When set to None, applies to all
                        asm register types.
                        Note: to apply to all data reg types ("freg","vreg",
                        "treg"), set this to None and set reg_type to data
    :type asm_reg_tag: str|None
    :param op1_type: first operation in the dependency chain link. When set to
                     None, applies to all operations
    :type op1_type: class:`Type[lsc_operation]|None`
    :param op2_type: second operation in the dependency chain link. When set
                     to None, applies to all operations
    :type op2_type: class:`Type[lsc_operation]|None`
    :param tr1_name: if the first operation is an lsc_transformation, this 
                     string specifies the exact transformation, i.e fma, dota,
                     fmul, etc... When set to None, applies to all 
                     transformations
    :type tr1_name: str|None
    :param tr2_name: if the first operation is an lsc_transformation, this 
                     string specifies the exact transformation, i.e fma, dota,
                     fmul, etc... When set to None, applies to all 
                     transformations
    :type tr2_name: str|None
    :param distance: distance, in independent instructions, to maintain.
    :type distance: int
    """
    dep_type     : dependency_type|None
    reg_type     : lsc_reg_type|None
    asm_reg_tag  : str|None
    op1_type     : Type[lsc_operation]|None
    op2_type     : Type[lsc_operation]|None
    tr1_name     : str|None
    tr2_name     : str|None
    distance     : int

    @classmethod
    def choose_from_enum(cls, choice : str, enum : Type[Enum]) -> Enum:
        result = None
        valid_values = [str(e).split('.')[-1] for e in enum]
        if not choice:
            pass
        elif choice not in valid_values:
            choices_str = ",".join([d.lower() for d in valid_values])
            raise ValueError(
                (f"Invalid spec: {choice}. Must be one of"
                 f" [{choices_str}]"
                 ))
        else:
            result = enum[choice]

        return result

    @classmethod
    def from_string(cls, specstr : str) -> distance_specification:
        """
        Format: dep:regt:rtag:op1:op2:dist
        """

        split_spec = specstr.split(":")

        dep_type = cls.choose_from_enum(choice=split_spec[0].upper(),
                                        enum=dependency_type)
        reg_type = cls.choose_from_enum(choice=split_spec[1],
                                        enum=lsc_reg_type)

        asm_reg_tag = None

        tag_str = split_spec[2]
        valid_tags = ["greg","freg","vreg","treg"]
        if tag_str:
            if tag_str not in valid_tags:
                choices_str = ",".join(valid_tags)
                raise ValueError(f"tag {tag_str} not in [{choices_str}]")
            asm_reg_tag = tag_str

        op1_str = split_spec[3]
        op2_str = split_spec[4]
        op_types = {
            "trf" : lsc_transformation,
            "fma" : lsc_transformation,
            "fmul" : lsc_transformation,
            "dota" : lsc_transformation,
            "opa" : lsc_transformation,
            "mma" : lsc_transformation,
            "ld" : lsc_load,
        "st" : lsc_store,
            "z" : lsc_zero,
            "aadd" : lsc_addr_add,
            "oadd" : lsc_add_val_off,
        }

        op1_type = None
        tr1_name = None
        op2_type = None
        tr2_name = None

        if op1_str:
            if op1_str not in op_types.keys():
                raise ValueError(f"Unknown op \"{op1_str}\"")
            op1_type = op_types[op1_str]
        if op2_str:
            if op2_str not in op_types.keys():
                raise ValueError(f"Unknown op \"{op2_str}\"")
            op2_type = op_types[op2_str]

        if op1_type == lsc_transformation:
            tr1_name = op1_str if "trf" != op1_str else None
        if op2_type == lsc_transformation:
            tr2_name = op2_str if "trf" != op2_str else None

        distance = int(split_spec[5])

        return  distance_specification(
                dep_type=dep_type, 
                reg_type=reg_type, 
                asm_reg_tag=asm_reg_tag,
                op1_type=op1_type, 
                op2_type=op2_type, 
                tr1_name=tr1_name, 
                tr2_name=tr2_name, 
                distance=distance)


    def apply(self,
              op1 : lsc_operation,
              op2 : lsc_operation) -> int:
        """
        Check if the distance constraint applies between the specified ops
        and if it does, returns the distance, otherwise returns 0
        """

        # Types

        if self.op1_type is not None and \
          not isinstance(op1, self.op1_type):
            return 0

        if self.op2_type is not None and \
          not isinstance(op2, self.op2_type):
            return 0
        
        # Exact transformations

        if self.op1_type == lsc_transformation:
            if self.tr1_name is not None:
                if self.tr1_name != op1.op:
                    return 0

        if self.op2_type == lsc_transformation:
            if self.tr2_name is not None:
                if self.tr2_name != op2.op:
                    return 0

        deps = get_dependencies(op1=op1, op2=op2)

        result = 0

        # registers
        for dep,regs in deps.items():
            if not regs:
                continue
            if self.dep_type is not None and self.dep_type != dep:
                continue
            for rc in regs:
                if self.reg_type is not None and self.reg_type != rc.ttype:
                    continue
                
                asm_reg_tag = determine_dreg_tag(dima=rc.t.dima, dimb=rc.t.dimb)
                if self.asm_reg_tag is not None and self.asm_reg_tag != asm_reg_tag:
                    continue
                    
                result = self.distance


        return result


