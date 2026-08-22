# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

"""
Structures for supporting different ways of moving data to/from memory and
between registers
"""

from __future__ import annotations

from enum import Enum,auto
from dataclasses import dataclass,field

from abc import abstractmethod, ABC

from asmgen.asmblocks.op import (
    operand_modifier as omod,
    operand_shape as osh,
    operation_signature as opsig,
    operation_modifier as opmod,
    register_type as rgt,
    make_ord_prefix as mop
)
from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import asm_data_type as adt

from ..matching.math import transformation as tf


class dm_direction(Enum):
    """
    Data movement direction:
    IN = input to arithmetic operation
    OUT = output of arithmetic operation
    """
    IN = auto()
    OUT = auto()

@dataclass(kw_only=True,frozen=True)
class register_reference:
    """
    Register reference for a data movement step, either reference to the
    operand of the arithmetic op or a temporary that requires allocation
    """
    is_op_opd : bool = False
    tag : str|None = None


    def __str__(self):
        if self.is_op_opd:
            return "$o"

        if self.tag is not None:
            return f"$o_{self.tag}"

        raise ValueError("Invalid register reference")

    def __repr__(self):
        return f"{self.__class__.__name__}('{str(self)}')"

    @classmethod
    def from_string(cls, s : str) -> "register_reference":
        """
        Create a register reference from a string
        """
        match s.split("_", 1):

            case ["$o"]:
                return orig_ref(is_op_opd=True, tag=None)

            case ["$o", tag]:
                return temp_ref(is_op_opd=False, tag=tag)

            case _:
                raise ValueError(f"String '{s}' is not a register reference")


@dataclass(kw_only=True,frozen=True)
class orig_ref(register_reference):
    """
    Reference to the "original" operand
    """
    is_op_opd : bool = True

@dataclass(kw_only=True,frozen=True)
class temp_ref(register_reference):
    """
    Reference to a temporary
    """
    is_op_opd : bool = False

@dataclass(kw_only=True)
class dm_step:
    """
    Data movement step as part of a transformation resolution
    """
    op:           str|None = None
    dest:         register_reference|None = None
    dest_rtype:   rgt|None = None
    src:          list[register_reference] = field(default_factory=list)
    src_rtypes:   list[rgt] = field(default_factory=list)
    op_mod_reqs:  set[opmod] = field(default_factory=set)
    opd_mod_reqs: dict[register_reference,set[omod]] = field(default_factory=dict)

@dataclass
class transformation_resolution:
    """
    "Resolution" aka one possible way to satisfy the data requirement of a
    transformed operand
    """
    unique_tag:   str
    steps:        list[dm_step]
    op_mod_reqs:  set[opmod] = field(default_factory=set)
    opd_mod_reqs: set[omod] = field(default_factory=set)



@dataclass(frozen=True)
class tfr_key:
    """
    The required transformation, register type and whether the operand is
    used as an input or output encode the "key" to the resolution registry
    """
    tfs: frozenset[tf]
    rtype: rgt
    ddir: dm_direction

class resolution_registry:
    """
    Registry for possible resolution of a transformation
    """

    def __init__(self):
        self.tfr_map : dict[tfr_key,list[transformation_resolution]] = {}
        self.tfr_tags : dict[tfr_key,set[str]] = {}


    def add_resolution(self, key : tfr_key, rsln : transformation_resolution):
        """
        Adds a possible resolution for a transformation
        """
        if key not in self.tfr_map:
            self.tfr_map[key] = []
            self.tfr_tags[key] = set()

        if rsln.unique_tag in self.tfr_tags[key]:
            raise ValueError("Duplicate unique_tag in added transformation_resolution")
        self.tfr_map[key].append(rsln)
        self.tfr_tags[key].add(rsln.unique_tag)


class resolution_provider(ABC):
    """
    Base class for providers of different transformation resolutions
    """
    @abstractmethod
    def register_resolutions(self, registry: resolution_registry):
        """
        Adds resolutions this object provides to a registry
        """

def filter_by_op_mods(sigs : list[opsig],
                      op_mod_reqs : set[opmod]
                      ) -> list[opsig]:
    """
    Filter signatures by selecting a subset that matches the modifier
    requirements
    """

    if op_mod_reqs:
        filtered_sigs : list[opsig] = []
        for sig in sigs:
            if op_mod_reqs.issubset(sig.modifiers):
                filtered_sigs.append(sig)
        return filtered_sigs

    return sigs

def filter_by_operand_mods(sigs : list[opsig],
                           opd_mod_reqs,
                           target_opd_name : str,
                           target_dt : adt,
                           target_rtype : rgt
                           ) -> list[opsig]:
    """
    Filter signatures by selecting a subset that matches the operand modifier
    requirements
    """

    if opd_mod_reqs:


        filtered_sigs = []
        for sig in sigs:
            if target_opd_name not in sig.operands:
                continue
            opd_sig = sig.operands[target_opd_name]
            if opd_sig.dt != target_dt:
                continue
            if opd_sig.rtype != target_rtype:
                continue
            if not opd_mod_reqs.issubset(opd_sig.modifiers):
                continue

            filtered_sigs.append(sig)

        return filtered_sigs

    return sigs


def get_op_rtype_req(steps : list[dm_step]) -> rgt:
    """
    extract register type requirement for the compute operation
    from the step list
    """
    for step in steps:
        if step.dest == orig_ref():
            return step.dest_rtype

        for src, rtype in zip(step.src, step.src_rtypes):
            if src == orig_ref():
                return rtype

    return None


def get_dm_sig_io_operands(opname : str,
                           sig : opsig) -> tuple[list[osh],list[osh]]:
    """
    Extract the inputs and outputs of a load/store/move operation
    """

    dregs = [sig.operands[name] for name in sig.operands if ('dreg' in name and '_' not in name)]


    if opname == "load":
        # all dregs are outputs (written to by instruction)
        return ([],dregs)
    if opname == "store":
        # all dregs are inputs (read from by instruction)
        return (dregs,[])
    if opname == "move":
        if len(dregs) == 2:
            return ([dregs[0]],[dregs[1]])

        input_count = sig.structural_params['nin']
        return (dregs[:input_count],dregs[input_count:])

    raise ValueError(f"Invalid dm op required: {opname}")


def filter_by_operand_count(
        opname : str,
        sigs : list[opsig],
        input_count : int,
        output_count : int
        ) -> list[opsig]:
    """
    Filters load/store/move operations by matching numbers of inputs/outputs
    """

    filtered_sigs = []
    for sig in sigs:
        inputs,outputs = get_dm_sig_io_operands(opname, sig)
        if len(inputs) != input_count or len(outputs) != output_count:
            continue

        filtered_sigs.append(sig)

    return filtered_sigs

def filter_by_step_requirements(
        sigs : list[opsig],
        step : dm_step,
        target_dt : adt,
        ) -> list[opsig]:
    """
    Filter data move (load/store/move) operation signatures by the requirements
    of the data move steps they are used in
    """

    # NOTE: having ONE target_dt excludes something like an
    #       fcvt z0.d, p0/m, z1.s. Maybe something to support in the future

    sigs = filter_by_op_mods(sigs, step.op_mod_reqs)
    if not sigs:
        return []

    # multiple outputs could possibly be handled here, but it's probably better
    # to handle it with an irmod, fusing multiple loads/stores/moves when
    # possible. Unless there is an exotic architecture that provides ONLY
    # instructions that work on multiple outputs
    output_count = 1
    input_count = len(step.src)

    sigs = filter_by_operand_count(step.op, sigs, input_count, output_count)
    if not sigs:
        return []

    dest_key = f"{mop(len(step.src))}dreg"

    opd_map = {f"{mop(i)}dreg" : ref for i,ref in enumerate(step.src)}
    opd_map[dest_key] = step.dest

    rtype_map = {f"{mop(i)}dreg" : rtype for i,rtype in enumerate(step.src_rtypes)}
    rtype_map[dest_key] = step.dest_rtype

    for opd,ref in opd_map.items():
        opd_mod_reqs = step.opd_mod_reqs.get(ref,set())
        rtype_req = rtype_map[opd]
        sigs = filter_by_operand_mods(sigs, opd_mod_reqs, opd,
                                  target_dt, rtype_req)

        if not sigs:
            break

    return sigs

def check_resolution(
        gen: asmgen,
        rsln: transformation_resolution,
        target_opd_name : str,
        target_dt : adt,
        target_op : str
        ) -> bool:
    """
    Check if a specific resolution is valid for the chosen operation,
    operand and data type
    """

    op = getattr(gen, target_op)
    if op is None:
        return False

    compute_sigs : list[opsig] = op.get_signatures()


    compute_sigs = filter_by_op_mods(compute_sigs, rsln.op_mod_reqs)
    if not compute_sigs:
        return False

    rtype_req = get_op_rtype_req(rsln.steps)
    if rtype_req is None:
        return False

    compute_sigs = filter_by_operand_mods(
            compute_sigs, rsln.opd_mod_reqs, target_opd_name,
            target_dt, rtype_req)
    if not compute_sigs:
        return False

    for step in rsln.steps:
        dmop = getattr(gen, step.op)
        if dmop is None:
            return False
        sigs = dmop.get_signatures()
        sigs = filter_by_step_requirements(sigs, step,
                                           target_dt)
        if not sigs:
            return False

    return True
