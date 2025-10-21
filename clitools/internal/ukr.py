# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

def get_ukr_components(ukr : str) -> list[str]:
    if "gemm" == ukr:
        return ["A","B","AB","C"]
    if "mm" == ukr:
        return ["A","B","C"]

    raise ValueError(f"Invalid microkernel {ukr}")
