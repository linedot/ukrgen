# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import os
import shutil
import unittest
import string
from git import Repo

from ukrgen.injectors.blis.patcher import blis_patcher

class test_blis_patcher(unittest.TestCase):
    def test_dead_end(self):

        bp = blis_patcher()

        with self.assertRaises(RuntimeError) as context:
            bp.prepare_code()
        self.assertTrue("Nothing new resolved" in str(context.exception))


    def test_prepare_code(self):

        bp = blis_patcher()

        configname="ukrgen_rvv"
        kernels = [
            ("GEMM","DOUBLE", f"bli_dgemm_${configname}_2vx12"),
        ]

        bp.prepare_code(params={
            "year" : "2025",
            "author" : "Stepan Nassyr <s.nassyr@fz-juelich.de>",
            "kernels" : kernels,
            "configname" : configname,
            "cntx_extra_defs" : "",
            "cntx_init_extra" : "",
            "archflags" : "-march=rv64gcv",
            "simd_size" : "32",
            "n" : "12",
            "kunroll" : "8",
            })

    def test_patch_blis(self):

        bp = blis_patcher()

        configname="ukrgen_rvv"
        kernels = [
            ("GEMM","DOUBLE", f"bli_dgemm_${configname}_2vx12"),
        ]

        bp.prepare_code(params={
            "year" : "2025",
            "author" : "Stepan Nassyr <s.nassyr@fz-juelich.de>",
            "kernels" : kernels,
            "configname" : configname,
            "cntx_extra_defs" : "",
            "cntx_init_extra" : "",
            "archflags" : "-march=rv64gcv",
            "simd_size" : "32",
            "n" : "12",
            "kunroll" : "8",
            })

        blis_orig_dir = "/tmp/blis-ukrgen-tests"
        if not os.path.isdir(blis_orig_dir):
            Repo.clone_from("https://github.com/flame/blis", blis_orig_dir)

        blis_patch_dir = "/tmp/blis-patchtest"
        if os.path.isdir(blis_patch_dir):
            shutil.rmtree(blis_patch_dir)

        bp.patch(blis_orig_dir, blis_patch_dir)


