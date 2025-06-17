# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from ...sequence_generator import sequence_generator

import abc

class trsm_mxn_register_sequence_generator(sequence_generator):
    required_parameters = [
        "a_data_register_list",
        "a_data_register_list_start",
        "b_data_register_list",
        "b_data_register_list_start",
        "m",
        "n"
    ]
    side_effects = [
    ]

    def initialize(self):
        self.a_data_register_list = self.internal_parameters["a_data_register_list"]
        if not (len(self.a_data_register_list) == len(set(self.a_data_register_list))):
            raise ValueError("non-unique-entries in a_data register list")
        self.a_data_register_list_position = self.internal_parameters["a_data_register_list_start"]

        self.b_data_register_list = self.internal_parameters["b_data_register_list"]
        if not (len(self.b_data_register_list) == len(set(self.b_data_register_list))):
            raise ValueError("non-unique-entries in b_data register list")
        self.b_data_register_list_position = self.internal_parameters["b_data_register_list_start"]

        self.c_data_register_list_position = self.b_data_register_list_position

        self.row_idx = 0

        self.unroll_idx = 0

        self.m = self.internal_parameters["m"]
        self.n = self.internal_parameters["n"]


        if len(self.b_data_register_list) != self.m*self.n:
            raise ValueError(f"C-tile register list size is not m*n ({self.m*self.n})")


    def advance(self):

        a_idx, b_idx, c_idx = self.calculate_indices()

        current_values={
            "a_vector" : self.a_data_register_list[a_idx],
            "b_vector" : self.b_data_register_list[b_idx],
            "c_vector" : self.b_data_register_list[c_idx],
        }

        a_idx = self.a_data_register_list_position
        b_idx = self.b_data_register_list_position
        c_idx = self.c_data_register_list_position

        a_idx, b_idx, c_idx = self.advance_simplified(
                a_idx=a_idx,
                b_idx=b_idx,
                c_idx=c_idx,
                )

        self.a_data_register_list_position = a_idx
        self.b_data_register_list_position = b_idx
        self.c_data_register_list_position = c_idx


        return current_values

    @abc.abstractmethod
    def calculate_indices(self) -> tuple[int,int,int]:
        raise NotImplementedError("Abstract method called")

    @abc.abstractmethod
    def advance_simplified(self, a_idx : int, b_idx : int, c_idx : int) -> tuple[int,int,int]:
        raise NotImplementedError("Abstract method called")


class trsm_l_rsg_topdown(trsm_mxn_register_sequence_generator):
    def calculate_indices(self) -> tuple[int,int,int]:

        a_idx = self.a_data_register_list_position + self.unroll_idx*(self.m*(self.m+1)//2)
        b_idx = self.b_data_register_list_position + (self.row_idx + self.unroll_idx*self.m)*self.n
        c_idx = self.c_data_register_list_position + (self.row_idx + self.unroll_idx*self.m)*self.n

        a_idx = a_idx % len(self.a_data_register_list)
        b_idx = b_idx % len(self.b_data_register_list)
        c_idx = c_idx % len(self.b_data_register_list)

        return a_idx, b_idx, c_idx

    def advance_simplified(self, a_idx : int, b_idx : int, c_idx : int) -> tuple[int,int,int]:

        #vectorization across n, unroll in m and m-i

        #example: 4x2v
        # fma_np (a, b, c) : c <- -(a*b)+c
        # fmul (a, b, c) : c <- a*b

        # matrices shown with size(v)=4
        #
        #      A                            B(m,:1v)                  B(m,1v:)
        # -------------------          -------------------       -------------------
        # | a11   0   0   0 |    b1v1> | b11 b12 b13 b14 | b1v2> | b15 b16 b17 b18 |
        # | a21 a22   0   0 |    b2v1> | b21 b22 b23 b24 | b2v2> | b25 b26 b27 b28 |
        # | a31 a32 a33   0 |    b3v1> | b31 b32 b33 b34 | b3v2> | b35 b36 b37 b38 |
        # | a41 a42 a43 a44 |    b4v1> | b41 b42 b43 b44 | b4v2> | b45 b46 b47 b48 |
        # -------------------          -------------------       -------------------

        # f0 < 1/a11
        # f1 < a21
        # f2 < a31
        # f3 < a41
        # v0 < b1v1
        # v1 < b1v2
        # v2 < b2v1
        # v3 < b2v2
        # v4 < b3v1
        # v5 < b3v2
        # v6 < b4v1
        # v7 < b4v2

        # fmul v0, f0, v0
        # fmul v1, f0, v1
        # At this point we can also store this row to C
        # v0 > c1v1
        # v1 > c1v2
        # 1st step of unrolled m-i loop:
        # fma_np v0, f1, v2  # < ~ i=1  ‾\
        # fma_np v1, f1, v3  # < ~ i=1    \   This can be reversed, increasing the
        # fma_np v0, f2, v4  # < ~ i=2     \  number of independent instructions
        # fma_np v1, f2, v5  # < ~ i=2     /  before the final fmuls
        # fma_np v0, f3, v6  # < ~ i=3    /   (could also reverse in last m-i step only)
        # fma_np v1, f3, v7  # < ~ i=3  _/

        # f4 < 1/a22
        # f5 < a32
        # f6 < a42
        # fmul v2, f4, v2  \ These can be moved up
        # fmul v3, f4, v3  /
        # v2 > c2v1
        # v3 > c2v2
        # 2nd step of unrolled m-i loop:
        # fma_np v2, f5, v4 # < ~ i=2
        # fma_np v3, f5, v5 # < ~ i=2
        # fma_np v2, f6, v6 # < ~ i=3
        # fma_np v3, f6, v7 # < ~ i=3

        # f7 < 1/a33
        # f8 < a43
        # fmul v4, f7, v4
        # fmul v5, f7, v5
        # v4 > c3v1
        # v5 > c3v2
        # 3rd step of unrolled m-i loop:
        # fma_np v4, f8, v6 # < ~ i=3
        # fma_np v5, f8, v7 # < ~ i=3

        # f9 < 1/a44
        # fmul v6, f9, v6
        # fmul v7, f9, v7
        # v6 > c4v1
        # v7 > c4v2

        c_idx = ((c_idx)+1) % ((self.m - self.row_idx)*self.n)
        if 0 == c_idx:
            self.row_idx = (self.row_idx + 1) % self.m

        # I don't think this unroll is valid, since there isn't a loop over k like in gemm?
        if (0 == c_idx) and (0 == self.row_idx):
            self.unroll_idx = self.unroll_idx + 1

        b_idx = ((b_idx+1) % self.n)
        if b_idx == 0:
            a_idx = ((a_idx+1) % len(self.a_data_register_list))


        return a_idx, b_idx, c_idx

class trsm_u_rsg_topdown(trsm_mxn_register_sequence_generator):
    def calculate_indices(self) -> tuple[int,int,int]:
        
        # position within the step
        a_idx = self.a_data_register_list_position - (self.row_idx*(self.m*2-self.row_idx+1)//2)
        # Bump up to last for this step
        a_idx += (self.m-self.row_idx-1)
        # Rotate around
        a_idx = a_idx % (self.m-self.row_idx)
        # bump up to absolute position
        a_idx += (self.row_idx*(self.m*2-self.row_idx+1)//2)
        # Bump up unroll
        a_idx += self.unroll_idx*(self.m*(self.m+1)//2)

        b_idx = self.b_data_register_list_position + ((self.m-self.row_idx-1) + self.unroll_idx*self.m)*self.n
        c_idx = self.c_data_register_list_position + ((self.m-self.row_idx-1) + self.unroll_idx*self.m)*self.n
        # rotate around 
        c_idx = c_idx % ((self.m-self.row_idx)*self.n)

        a_idx = a_idx % len(self.a_data_register_list)
        b_idx = b_idx % len(self.b_data_register_list)
        c_idx = c_idx % len(self.b_data_register_list)

        return a_idx, b_idx, c_idx
    def advance_simplified(self, a_idx : int, b_idx : int, c_idx : int) -> tuple[int,int,int]:

        #vectorization across n, unroll in m and m-i

        #example: 4x2v
        # fma_np (a, b, c) : c <- -(a*b)+c
        # fmul (a, b, c) : c <- a*b

        # matrices shown with size(v)=4
        #
        #      A                            B(m,:1v)                  B(m,1v:)
        # -------------------          -------------------       -------------------
        # | a11 a12 a13 a14 |    b1v1> | b11 b12 b13 b14 | b1v2> | b15 b16 b17 b18 |
        # |   0 a22 a23 a24 |    b2v1> | b21 b22 b23 b24 | b2v2> | b25 b26 b27 b28 |
        # |   0   0 a33 a34 |    b3v1> | b31 b32 b33 b34 | b3v2> | b35 b36 b37 b38 |
        # |   0   0   0 a44 |    b4v1> | b41 b42 b43 b44 | b4v2> | b45 b46 b47 b48 |
        # -------------------          -------------------       -------------------

        # f0 < a14
        # f1 < a24
        # f2 < a34
        # f3 < 1/a44

        # v0 < b1v1
        # v1 < b1v2
        # v2 < b2v1
        # v3 < b2v2
        # v4 < b3v1
        # v5 < b3v2
        # v6 < b4v1
        # v7 < b4v2

        # fmul v6, f3, v6
        # fmul v7, f3, v7
        # At this point we can also store this row to C
        # v6 > c4v1
        # v7 > c4v2
        # 1st step of unrolled m-i loop:
        # fma_np v6, f0, v0  # < ~ i=1  ‾\
        # fma_np v7, f0, v1  # < ~ i=1    \   This can be reversed, increasing the
        # fma_np v6, f1, v2  # < ~ i=2     \  number of independent instructions
        # fma_np v7, f1, v3  # < ~ i=2     /  before the final fmuls
        # fma_np v6, f2, v4  # < ~ i=3    /   (could also reverse in last m-i step only)
        # fma_np v7, f2, v5  # < ~ i=3  _/

        # f4 < a13
        # f5 < a23
        # f6 < 1/a33
        # fmul v4, f6, v4  \ These can be moved up
        # fmul v5, f6, v5  /
        # v4 > c3v1
        # v5 > c3v2
        # 2nd step of unrolled m-i loop:
        # fma_np v4, f4, v0 # < ~ i=2
        # fma_np v5, f4, v1 # < ~ i=2
        # fma_np v4, f5, v2 # < ~ i=3
        # fma_np v5, f5, v3 # < ~ i=3

        # f7 < a12
        # f8 < 1/a22
        # fmul v2, f8, v2
        # fmul v3, f8, v3
        # v2 > c2v1
        # v3 > c2v2
        # 3rd step of unrolled m-i loop:
        # fma_np v2, f7, v0 # < ~ i=3
        # fma_np v3, f7, v1 # < ~ i=3

        # f9 < 1/a11
        # fmul v0, f9, v0
        # fmul v1, f9, v1
        # v6 > c1v1
        # v7 > c1v2

        c_idx = ((c_idx)+1) % ((self.m - self.row_idx)*self.n)
        if 0 == c_idx:
            self.row_idx = (self.row_idx + 1) % self.m

        # I don't think this unroll is valid, since there isn't a loop over k like in gemm?
        if (0 == c_idx) and (0 == self.row_idx):
            self.unroll_idx = self.unroll_idx + 1

        b_idx = ((b_idx+1) % self.n)
        if b_idx == 0:
            a_idx = ((a_idx+1) % len(self.a_data_register_list))


        return a_idx, b_idx, c_idx
