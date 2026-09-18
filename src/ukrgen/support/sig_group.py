
# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

"""
Grouping signatures that are representing equivalent hw implementations with support for different addressing modes
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass


from asmgen.registers import asm_data_type as adt

from asmgen.asmblocks.op import (
    operation_signature as opsig,
    operation_modifier as opmod,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rgt,
    opdna1_modifier as dna1mod,
    opd3_modifier as d3mod,
    move_modifier as mvmod,
    operand_modifier as opdmod
)


IDENTITY_ROLES = frozenset({orl.DATA})


NONID_OP_MODS = frozenset({
    mod for mod in dna1mod if mod not in {dna1mod.STRUCT}
    })

NONID_OP_MODS |= {d3mod.MASK, mvmod.MASK}


NONID_STRUCT_PARAMS = frozenset({"it"})


SEMANTIC_OP_MODS = frozenset({
    d3mod.NA, d3mod.NP, d3mod.NX, d3mod.MULC
    })


def is_identity_operand(oshape : osh) -> bool:
    """
    Checks whether the operand shape describes an operand that
    is part of the operation identity

    :param oshape: operand shape to check
    :return: True if operand part of identity, False otherwise
    """
    return oshape.otype == ot.REGISTER and oshape.orole in IDENTITY_ROLES


def identity_mods(sig: opsig) -> frozenset[opmod]:
    """
    Extract operation modifiers that are part of the operation identity from
    the signature

    :param sig: operation signature to extract modifiers from
    :return: set of identity-relevant modifiers the signature had
    """

    return frozenset(sig.modifiers) - NONID_OP_MODS

@dataclass(kw_only=True,frozen=True)
class sig_identity:
    """
    Structure representing the identity of the signature
    """
    opmods : frozenset[opmod]
    operands : tuple[tuple[str,rgt,adt,frozenset[opdmod]],...]
    struct : frozenset[tuple[str,Any]] # can be int, can be enum value...

def get_sig_identity(sig: opsig) -> sig_identity:
    """
    Determine the identity of an operation signature

    :param sig: Operation signature to identify
    :return: Identity of the signature that can be used as a key
    """
    opds = tuple(sorted(
        (name, shape.rtype, shape.dt, 
         frozenset(shape.modifiers))
        for name, shape in sig.operands.items()
        if is_identity_operand(shape)
    ))

    struct = frozenset(
        (k,v) for k, v in sig.structural_params.items()
        if k not in NONID_STRUCT_PARAMS
    )

    return sig_identity(opmods=identity_mods(sig), operands=opds, struct=struct)


def consumed_registers(sig: opsig) -> dict[rgt, int]:
    """
    Count how many of which register type this operation consumes

    :param sig: Operation signature to count consumed register for
    :return: Numbers of consumed register mapped onto register types
    """


    counts : dict[rgt,int] = {}

    for shape in sig.operands.values():
        if shape.otype != ot.REGISTER or shape.rtype is None:
            continue

        counts[shape.rtype] = counts.get(shape.rtype, 0)+1
    return counts


@dataclass(frozen=True)
class sig_group:
    """
    Signature group, containing multiple signatures that share an identity
    """

    ident : sig_identity
    sigs: tuple[opsig,...]


    def __len__(self) -> int:
        return len(self.sigs)


    @property
    def representative(self) -> opsig:
        """
        Get a representative signature for the whole group
        """
        
        return self.sigs[0]



    def intersect(self, other : sig_group) -> sig_group|None:
        """
        Intersect two signature groups of the same identity

        :param other: Other signature group to intersect with
        :return: intersected group or None if the intersection is empty
                 or the identity doesn't match
        """

        if self.ident != other.ident:
            return None

        keep = {id(s) for s in other.sigs}
        sigs = tuple(s for s in self.sigs if id(s) in keep)
        if not sigs:
            return None

        return sig_group(ident=self.ident, sigs=sigs)

    def describe(self) -> str:
        """
        Human-readable description of the group
        """

        mods = sorted(m.name for m in self.ident.opmods)
        opds = self.ident.operands
        struct = self.ident.struct


        parts = []

        if mods:
            parts.append("+".join(sorted(mods)))

        for name, rtype, _dt, opd_mods in opds:
            opd_mod_names = sorted(m.name for m in opd_mods)
            tag = name if not opd_mods else f"{name}[{'+'.join(sorted(opd_mod_names))}]"
            parts.append(f"{tag}:{rtype.name if rtype is not None else 'None'}")
        for k,v in sorted(struct,key=lambda kv: str(kv[0])):
            parts.append(f"{k}={v}")
        return ",".join(parts)


    def __str__(self) -> str:
        return f"<{self.describe()} ({len(self.sigs)} sigs)>"

    def __repr__(self) -> str:
        return str(self)


def group_sigs(sigs: list[opsig]) -> list[sig_group]:
    """
    Partition signatures into signature groups


    :param sigs: List of signatures to partition
    :return: list of signature groups containing all the original signatures
    """

    groups : dict[sig_identity, list[opsig]] = {}

    for sig in sigs:
        groups.setdefault(get_sig_identity(sig), []).append(sig)


    return [sig_group(ident=ident, sigs=tuple(members))
            for ident, members in sorted(groups.items(),
                                         key=lambda kv: str(kv[0]))
    ]

def filter_by_forbidden_semantics(sigs: list[opsig],
                                  allowed: frozenset[opmod] = frozenset()
                                  ) -> list[opsig]:
    """
    Filter signatures by modifiers that change the computation

    :param sigs: List of signatures to filter
    :param allowed: Allowed semantical modifiers
    :return: filtered list of signatures
    """

    forbidden = SEMANTIC_OP_MODS - allowed

    if not forbidden:
        return sigs

    return [sig for sig in sigs
            if not forbidden & frozenset(sig.modifiers)]
