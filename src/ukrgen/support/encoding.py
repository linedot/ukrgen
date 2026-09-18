# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

"""
Textual encoding of hardware implementations.

Each implementation the solver finds gets one string, and that string is what the user
passes back on the command line to pin a choice (--mm-to, --scale-to, --store-to, ...).
The encoding therefore has to be injective: two implementations that emit different
assembly must never share a string, or the choice is ambiguous.

Grammar of one operation:

    opname[discriminant](opd, opd, ...){index map}

  discriminant  what distinguishes this signature group from the others of the same
                operation - operation modifiers, operand modifiers and structural
                parameters. Omitted when the operation has only one group.
  opd           <math name>.<register type>:<resolutions>, in hardware slot order
                (adreg, bdreg, cdreg, ...), so the slot is implied by position.
                Resolutions are <in> for an input, >:<out> for an output, and
                <in>>:<out> for an operand that is both.
  index map     hardware index = requirement index, sorted. Requirement indices that
                appear in no hardware operand are extent 1 and are listed as
                <idx>=1 for readability.

Chains of several operations are joined with " ; " and share temporary names (T0, ...).
"""

from __future__ import annotations

from asmgen.asmblocks.op import operand_type as ot, operand_role as orl

from .data_move import dm_direction as dmd
from .dm_solver import operation_resolution, resolved_operation_chain
from .sig_group import sig_group

from ..matching.math import (
    ast_node,
    recurse_indices,
    reduction_dims,
    req_solution_step,
)


def slot_order(sig_grp: sig_group) -> list[str]:
    """
    Hardware data operand names in signature order (adreg, bdreg, cdreg, ...).

    :param sig_grp: signature group of the operation
    :return: data operand names, ordered
    """
    sig = sig_grp.representative
    names = [name for name, shape in sig.operands.items()
             if shape.otype == ot.REGISTER and shape.orole == orl.DATA]

    return sorted(names)


def group_discriminant(sig_grp: sig_group, all_groups: list[sig_group]) -> str:
    """
    Shortest tag distinguishing one signature group from the others of the same operation.

    :param sig_grp: the group to describe
    :param all_groups: every group of that operation, to tell whether a tag is needed
    :return: discriminant string, empty when the operation has only one group
    """
    if len(all_groups) < 2:
        return ""

    return sig_group_tag(sig_grp)


def sig_group_tag(sig_grp: sig_group) -> str:
    """
    Compact tag naming what distinguishes a signature group: operation modifiers,
    operand modifiers and structural parameters. Register types are left out, they
    already appear in the operand part of the encoding.

    :param sig_grp: group to tag
    :return: tag, empty when the group carries no distinguishing feature
    """
    parts = sorted(m.name for m in sig_grp.ident.opmods)

    for name, _rtype, _dt, opd_mods in sig_grp.ident.operands:
        for m in sorted(om.name for om in opd_mods):
            parts.append(f"{name}:{m}")

    for k, v in sorted(sig_grp.ident.struct, key=lambda kv: str(kv[0])):
        val = v.name if hasattr(v, "name") else v
        parts.append(f"{k}={val}")

    return "+".join(parts)


def steps_tag(strat, discriminating: set[tuple[str,int]] | None = None) -> str:
    """
    Tag naming the signature groups chosen for an operand's data movement steps.

    Two strategies can share a resolution tag and still differ per step - bc_scalar
    resolved with a masked mov is not the same implementation as one resolved with an
    unmasked dup - so the step groups have to appear in the encoding. Steps whose group
    is the same in every candidate carry no information and are left out: on SVE every
    load and store is predicated, so printing MASK on each of them only adds noise.

    :param strat: resolved operand strategy
    :param discriminating: (resolution tag, step index) pairs that actually vary across
                           the candidate list. Pass None to tag every step.
    :return: tag like "[1:MASK]", empty when no step distinguishes anything
    """
    parts = []
    for i, rstep in enumerate(strat.resolved_steps):
        if discriminating is not None and (strat.rsln.unique_tag, i) not in discriminating:
            continue
        tag = sig_group_tag(rstep.sg)
        if tag:
            parts.append(f"{i}:{tag}")

    if not parts:
        return ""

    return "[" + ",".join(parts) + "]"


def discriminating_steps(impls: list[resolved_operation_chain]) -> set[tuple[str,int]]:
    """
    Find the data movement steps whose signature group varies between candidates.

    :param impls: implementations to scan
    :return: set of (resolution tag, step index) that take more than one group
    """
    seen : dict[tuple[str,int], set[str]] = {}
    for impl in impls:
        for res in impl.resolved_chain:
            for strat in res.operand_strategies.values():
                for i, rstep in enumerate(strat.resolved_steps):
                    seen.setdefault((strat.rsln.unique_tag, i), set()).add(
                            sig_group_tag(rstep.sg))

    return {key for key, tags in seen.items() if len(tags) > 1}


def encode_operation(res: operation_resolution,
                     step: req_solution_step,
                     req_indices: list[str],
                     red_dims: set[str],
                     all_groups: list[sig_group] | None = None,
                     discriminating: set[tuple[str,int]] | None = None) -> str:
    """
    Encode one operation of an implementation.

    :param res: resolved operation
    :param step: matching solution step, for the name and index mappings
    :param req_indices: every index of the requirement, in a stable order
    :param red_dims: requirement indices that are reduced over
    :param all_groups: every signature group of this operation, to decide whether the
                       discriminant is needed. Pass None to always emit it.
    :param discriminating: data movement steps whose group varies between candidates
    :return: encoded operation
    """
    by_name : dict[str, dict[dmd, str]] = {}
    for (opd_name, ddir), strat in res.operand_strategies.items():
        by_name.setdefault(opd_name, {})[ddir] = \
            f"{strat.rsln.unique_tag}{steps_tag(strat, discriminating)}"

    sig = res.compute_sg.representative

    opd_strs = []
    for hw_name in slot_order(res.compute_sg):
        if hw_name not in by_name:
            continue
        math_name = step.name_mapping.get(hw_name, hw_name)
        rtype = sig.operands[hw_name].rtype
        rtype_str = rtype.name if rtype is not None else "?"

        dirs = by_name[hw_name]
        if dmd.IN in dirs and dmd.OUT in dirs:
            rsln_str = f"{dirs[dmd.IN]}>{dirs[dmd.OUT]}"
        elif dmd.OUT in dirs:
            rsln_str = f">{dirs[dmd.OUT]}"
        else:
            rsln_str = dirs[dmd.IN]

        opd_strs.append(f"{math_name}.{rtype_str}:{rsln_str}")

    idx_parts = []
    mapped = {r: h for h, r in step.index_mapping.items()}
    for r_idx in req_indices:
        tag = "*" if r_idx in red_dims else ""
        if r_idx in mapped:
            idx_parts.append(f"{mapped[r_idx]}={r_idx}{tag}")
        else:
            idx_parts.append(f"{r_idx}{tag}=1")

    disc = group_discriminant(res.compute_sg, all_groups) if all_groups is not None \
        else sig_group_tag(res.compute_sg)
    disc_str = f"[{disc}]" if disc else ""

    return f"{res.opname}{disc_str}({', '.join(opd_strs)})" \
           f"{{{', '.join(idx_parts)}}}"


def encode_implementation(impl: resolved_operation_chain,
                          req: ast_node,
                          groups_by_op: dict[str, list[sig_group]] | None = None,
                          discriminating: set[tuple[str,int]] | None = None) -> str:
    """
    Encode a whole implementation as the string the user passes back to select it.

    :param impl: implementation to encode
    :param req: the requirement it implements, for the index list
    :param groups_by_op: every signature group per operation name, so the discriminant
                         can be left out where an operation has only one group
    :return: encoded implementation
    """
    req_indices = list(recurse_indices(req))
    red_dims = reduction_dims(req)
    for d in sorted(red_dims):
        if d not in req_indices:
            req_indices.append(d)

    parts = []
    for res, step in zip(impl.resolved_chain, impl.math_chain):
        groups = None if groups_by_op is None else groups_by_op.get(res.opname)
        parts.append(encode_operation(res, step, req_indices, red_dims,
                                      groups, discriminating))

    return " ; ".join(parts)


def collect_groups(impls: list[resolved_operation_chain]) -> dict[str, list[sig_group]]:
    """
    Gather the distinct signature groups used per operation across a set of
    implementations, so the encoder can drop discriminants that distinguish nothing.

    :param impls: implementations to scan
    :return: operation name -> list of distinct signature groups
    """
    groups : dict[str, list[sig_group]] = {}
    for impl in impls:
        for res in impl.resolved_chain:
            known = groups.setdefault(res.opname, [])
            if not any(g.ident == res.compute_sg.ident for g in known):
                known.append(res.compute_sg)

    return groups


def encode_all(impls: list[resolved_operation_chain],
               req: ast_node) -> list[str]:
    """
    Encode every implementation of a requirement.

    :param impls: implementations to encode
    :param req: the requirement they implement
    :return: one string per implementation, in the order given
    """
    groups_by_op = collect_groups(impls)
    discriminating = discriminating_steps(impls)

    return [encode_implementation(impl, req, groups_by_op, discriminating)
            for impl in impls]
