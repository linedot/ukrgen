# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

"""
Solving data movement requirements of possible computational solutions.
"""

import itertools

from dataclasses import dataclass, field

from asmgen.asmblocks.noarch import asmgen
from asmgen.asmblocks.op import (
    operation_signature as opsig,
    register_type as rgt
)
from asmgen.registers import (
    asm_data_type as adt,
    adt_is_float,
    adt_is_int
)
from .data_move import (
    dm_step,
    dm_direction as dmd,
    resolution_registry,
    transformation_resolution as tr,
    tfr_key,
    filter_by_op_mods,
    filter_by_operand_mods,
    filter_by_step_requirements,
    filter_by_forbidden_op_mods,
    filter_by_forbidden_operand_mods,
    get_op_rtype_req
)

from ..matching.math import (
    transformation as tf,
    ast_node,
    get_operand_io,
    req_solution_step as rsstep,
    for_each_operand
)

from .sig_group import (
    sig_group,
    group_sigs,
    filter_by_forbidden_semantics
)


@dataclass(kw_only=True)
class resolved_dm_step:
    """
    data movement step with signatures that fulfill its requirements

    :param step: data movement step
    :param sg: signature group selected from those provided by the currently used
               generator that fulfills the requirements of the step
    """
    step: dm_step
    sg: sig_group


@dataclass(kw_only=True)
class resolved_operand_strategy:
    """
    Transformation resolution with compute signatures that fulfill it's requirements and
    resolved data movement steps

    :param rsln: transformation resolution
    :param compute_sg: Compute operation signature group selected from sigs provided by
                       the currently used generator that fulfill the requirements of
                       the resolution
    :param resolved_steps: list of resolved steps, in order, to fulfill the operand
                           requirements of the resolution
    """

    rsln : tr
    compute_sg: sig_group
    resolved_steps: list[resolved_dm_step]


@dataclass(kw_only=True)
class operation_resolution:
    """
    Fully encodes a possible usage of the compute operation with at least one valid resolution
    for moving data into/out of it's operands

    :param opname: name of the compure operation
    :param compute_sg: Group of compute operation signatures supporting this usage
    :param operand_strategies: operand resolutions mapped to operand names and data movement
                               direction as they are used by the compute operation
    """

    opname : str
    compute_sg: sig_group
    operand_strategies: dict[tuple[str, dmd], resolved_operand_strategy]


    def describe(self) -> str:
        """
        human-readable description for this operation usage
        """
        opds = ", ".join(
            f"{name}:{ddir.name}={strat.rsln.unique_tag}"
            for (name,ddir), strat in sorted(
                self.operand_strategies.items(),
                key=lambda kv: (kv[0][0],kv[0][1].name)
            )
        )
        return f"{self.opname}[{self.compute_sg.describe()}]({opds})"


# Tried splitting it up. made it less readable
# pylint: disable-next=too-many-locals
def generate_operand_resolution_candidates(
        gen : asmgen,
        rslns : list[tr],
        target_opd_name: str,
        target_dt: adt,
        compute_sg: sig_group
        ) -> list[resolved_operand_strategy]:
    """
    Select valid transformation resolution candidates for a single operand of a compute
    operation

    :param gen: Generator to query for operation signatures
    :param rslns: Transformation resolutions to select from
    :param target_opd_name: operand name as used in the compute operation
    :param target_dt: operand data type
    :param compute_sg: compute signature group this operand is being resolved against
    :return: list of strategies for this operand
    """

    candidates : list[resolved_operand_strategy] = []

    for rsln in rslns:


        compute_sigs = filter_by_op_mods(list(compute_sg.sigs), rsln.op_mod_reqs)
        compute_sigs = filter_by_forbidden_op_mods(compute_sigs, 
                                                   rsln.forbidden_op_mods)
        if not compute_sigs:
            continue

        rtype_req = get_op_rtype_req(rsln.steps)
        if rtype_req is None:
            continue

        compute_sigs = filter_by_operand_mods(
                compute_sigs, rsln.opd_mod_reqs, target_opd_name,
                target_dt, rtype_req)
        compute_sigs = filter_by_forbidden_operand_mods(
                compute_sigs, rsln.forbidden_opd_mods, target_opd_name)
        if not compute_sigs:
            continue

        narrowed_sg = sig_group(ident=compute_sg.ident, sigs=tuple(compute_sigs))

        step_groups : list[list[sig_group]] = []
        step_impossible = False
        for step in rsln.steps:
            dmop = getattr(gen, step.op, None)
            if dmop is None:
                step_impossible = True
                break
            sigs = filter_by_step_requirements(
                    dmop.get_signatures(), step, target_dt)
            if not sigs:
                step_impossible = True
                break


            step_groups.append(group_sigs(sigs))

        if step_impossible:
            continue


        for combo in itertools.product(*step_groups):
            resolved_steps = [
                resolved_dm_step(step=step, sg=grp)
                for step, grp in zip(rsln.steps, combo)
            ]

        candidates.append(
                resolved_operand_strategy(
                    rsln=rsln,
                    compute_sg=narrowed_sg,
                    resolved_steps=resolved_steps))

    return candidates


def deduce_operand_rtype(
        ast : ast_node,
        opd_name : str,
        dt : adt
        ) -> rgt:
    """
    Find out what register type is required for an operand in the AST by investigating
    it's dimensions

    :param ast: AST that uses this operand
    :param opd_name: Name of the operand as it is used in the AST
    :param dt: Data type (to distinguish between FP and GP)
    :return: Required register type
    """

    dim_count = None
    for opd in for_each_operand(ast, lambda x: x):
        if opd.name == opd_name:
            dim_count = sum(1 for idx in opd.indices if idx is not None)

    if dim_count is None:
        raise ValueError(f"Operand {opd_name} not used in AST {ast}")

    if dim_count == 2:
        return rgt.TILE
    if dim_count == 1:
        return rgt.VEC
    if dim_count == 0:
        if adt_is_float(dt):
            return rgt.FP
        if adt_is_int(dt):
            return rgt.GP

        raise ValueError(f"Invalid data type {dt}")

    raise ValueError(f"Invalid number of dimensions in operand {opd_name}: {dim_count}")


def get_opd_candidates(*,
        gen : asmgen,
        tfs : dict[str,tf],
        dir_reqs : dict[str,set[dmd]],
        hw_dts : dict[str,adt],
        hw_rtypes : dict[str,rgt],
        compute_sg : sig_group,
        registry : resolution_registry
        ) -> dict[tuple[str,dmd], list[resolved_operand_strategy]]:
    """
    For each operand and direction, get a list of valid strategies against one
    compute signature group

    :param gen: Generator to inspect the operations of
    :param tfs: required operand transformations
    :param dir_reqs: Operand I/O role in the operation (input and/or output)
    :param hw_dts: operand data types
    :param hw_rtypes: operand register types
    :param compute_sg: compute signature group to resolve the operands against
    :param registry: Registry containing available operand transformation resolutions
    :return: list of possible operand strategies for each operand and dm direction
    """

    opd_candidates : dict[tuple[str,dmd],list[resolved_operand_strategy]] = {}

    for opd_name, opd_tfs in tfs.items():

        for ddir in dir_reqs.get(opd_name,set()):
            key = tfr_key(tfs=frozenset(opd_tfs), rtype=hw_rtypes[opd_name], ddir=ddir)

            rslns = registry.tfr_map.get(key, [])

            candidates = generate_operand_resolution_candidates(
                    gen=gen,
                    rslns=rslns,
                    target_opd_name=opd_name,
                    target_dt=hw_dts[opd_name],
                    compute_sg=compute_sg)

            if not candidates:
                return {}

            opd_candidates[(opd_name, ddir)] = candidates


    return opd_candidates


def enumerate_resolutions(*,
        gen : asmgen,
        tfs : dict[str,tf],
        dir_reqs : dict[str,set[dmd]],
        hw_dts : dict[str,adt],
        hw_rtypes : dict[str,rgt],
        opname : str,
        registry : resolution_registry,
        allowed_semantics : frozenset[opmod] = frozenset()
        ) -> list[operation_resolution]:
    """
    Given an operation, operand names, transformations, directions, data and register
    types, select and resolve valid operation usages

    :param gen: Generator to inspect the operations of
    :param tfs: required operand transformations
    :param dir_reqs: Operand I/O role in the operation (input and/or output)
    :param hw_dts: operand data types
    :param hw_rtypes: operand register types
    :param opname: name of the operation
    :param registry: Registry containing available operand transformation resolutions
    :param allowed_semantics: Semantic operation modifiers the requirement asks for
    :return: list of operation resolutions that can be used to satisfy the requirements
    """


    op = getattr(gen, opname, None)
    if op is None:
        return []


    all_sigs = filter_by_forbidden_semantics(op.get_signatures(), allowed_semantics)
    if not all_sigs:
        return []

    valid_resolutions = []

    for compute_sg in group_sigs(all_sigs):

        opd_candidates = get_opd_candidates(
                gen=gen, tfs=tfs, dir_reqs=dir_reqs, hw_dts=hw_dts,
                hw_rtypes=hw_rtypes, compute_sg=compute_sg,
                registry=registry
            )
        
        combo_keys = list(opd_candidates.keys())
        if not combo_keys:
            continue

        for combo in itertools.product(*(opd_candidates[k] for k in combo_keys)):
            combo_dict = dict(zip(combo_keys, combo))

            narrowed_sg = combo_dict[combo_keys[0]].compute_sg
            for k in combo_keys[1:]:
                narrowed_sg = narrowed_sg.intersect(combo_dict[k].compute_sg)
                if narrowed_sg is None:
                    break
            if narrowed_sg is None:
                continue
            valid_resolutions.append(
                    operation_resolution(
                        opname=opname,
                        compute_sg=narrowed_sg,
                        operand_strategies=combo_dict
                        )
                    )

    return valid_resolutions


def get_dir_reqs(ast : ast_node, opds : list[str]) -> dict[str,set[dmd]]:
    """
    Given a list of operand names and an AST, determine for each operand if they are used as
    inputs to, or outputs of, the AST - or both and return a dictionary with the information

    :param ast: AST to query
    :param opds: list of operand names
    :return: data movement directions as a set mapped to each operand name in the list
    """
    dir_reqs : dict[str, set[dmd]] = {}
    for opd_name in opds:
        is_in, is_out = get_operand_io(ast, opd_name)
        dirs = set()
        if is_in:
            dirs.add(dmd.IN)
        if is_out:
            dirs.add(dmd.OUT)

        dir_reqs[opd_name] = dirs

    return dir_reqs


def resolve_ast_solution(
        gen: asmgen,
        solution_step: rsstep,
        dts: dict[str,adt],
        target_op: str,
        registry: resolution_registry
        ) -> list[operation_resolution]:
    """
    Given one step of a solution to a requirement AST, determine hw requirements and find
    supported compute operations and data movement operations that enable its usage

    :param gen: Generator to query for operations
    :param solution_step: Step of the solution of the AST requirement
    :param dts: Data types of the operands as used in the solution AST
    :param target_op: Compute operation to query support for
    :param registry: Resolution registry for transformed operands
    :return: list of valid resolutions for this operation and solution step
    """

    hw_ast: ast_node            = solution_step.hw_ast
    opd_tfs: dict[str,set[tf]]  = solution_step.transformations
    name_mapping: dict[str,str] = solution_step.name_mapping


    hw_dts : dict[str,adt] = {}
    for hw_name, req_name in name_mapping.items():
        if req_name in dts:
            hw_dts[hw_name] = dts[req_name]
        else:
            raise ValueError(f"No data type for {req_name} specified")

    hw_rtypes : dict[str,rgt] = {}
    for opd_name in opd_tfs.keys():
        hw_rtypes[opd_name] = deduce_operand_rtype(hw_ast, opd_name, hw_dts[opd_name])

    dir_reqs : dict[str, set[dmd]] = get_dir_reqs(hw_ast, opd_tfs.keys())


    return enumerate_resolutions(
            gen=gen,
            tfs=opd_tfs,
            dir_reqs=dir_reqs,
            hw_dts=hw_dts,
            hw_rtypes=hw_rtypes,
            opname=target_op,
            registry=registry)



@dataclass(kw_only=True)
class resolved_operation_chain:
    """
    Complete hardware implementation equivalent to the solution of an AST requirement
    """

    math_chain: list[rsstep] = field(default_factory=list)
    resolved_chain: list[operation_resolution] = field(default_factory=list)


    def encode(self) -> list[str]:
        """
        Returns a string representation of the chain
        """
        step_encodings = []
        for res in self.resolved_chain:
            opd_str = []
            for (opd_name, ddir), strat in res.operand_strategies.items():
                opd_str.append(f"{opd_name}:{ddir.name}={strat.rsln.name}")

            step_encodings.append(f"{res.opname}({', '.join(opd_str)})")


    @classmethod
    def decode(cls, step_encodings : str, registry : resolution_registry):
        """
        constructs the chain from its string representation
        """
