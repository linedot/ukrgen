from ...sequence_generator import sequence_generator

import abc

class mm_mxn_register_sequence_generator(sequence_generator):
    required_parameters = [
        "a_data_register_list",
        "a_data_register_list_start",
        "b_data_register_list",
        "b_data_register_list_start",
        "c_data_register_list",
        "c_data_register_list_start",
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

        self.c_data_register_list = self.internal_parameters["c_data_register_list"]
        if not (len(self.c_data_register_list) == len(set(self.c_data_register_list))):
            raise ValueError("non-unique-entries in c_data register list")
        self.c_data_register_list_position = self.internal_parameters["c_data_register_list_start"]

        self.a_idx_offset = 0
        self.b_idx_offset = 0

        self.m = self.internal_parameters["m"]
        self.n = self.internal_parameters["n"]


        if len(self.c_data_register_list) != self.m*self.n:
            raise ValueError(f"C-tile register list size is not m*n ({self.m*self.n})")


    def advance(self):

        a_idx = (self.a_data_register_list_position + self.a_idx_offset) % len(self.a_data_register_list)
        b_idx = (self.b_data_register_list_position + self.b_idx_offset) % len(self.b_data_register_list)
        c_idx = self.c_data_register_list_position

        current_values={
            "a_vector" : self.a_data_register_list[a_idx],
            "b_vector" : self.b_data_register_list[b_idx],
            "c_vector" : self.c_data_register_list[c_idx],
        }

        a_idx = self.a_data_register_list_position
        b_idx = self.b_data_register_list_position
        c_idx = self.c_data_register_list_position

        a_idx, b_idx, c_idx = self.advance_simplified(
                a_idx=a_idx,
                b_idx=b_idx,
                c_idx=c_idx
                )

        self.a_data_register_list_position = a_idx
        self.b_data_register_list_position = b_idx
        self.c_data_register_list_position = c_idx


        return current_values


    @abc.abstractmethod
    def advance_simplified(self, a_idx : int, b_idx : int, c_idx : int) -> tuple[int,int,int]:
        raise NotImplementedError("Abstract method called")


class mm_rsg_afirst(mm_mxn_register_sequence_generator):
    def advance_simplified(self, a_idx : int, b_idx : int, c_idx : int) -> tuple[int,int,int]:

        c_idx = ((c_idx + 1) % (self.m*self.n))
        if 0 == c_idx:
            self.a_idx_offset += self.m
            self.a_idx_offset = self.a_idx_offset % len(self.a_data_register_list)
            self.b_idx_offset += self.n
            self.b_idx_offset = self.b_idx_offset % len(self.b_data_register_list)
        a_idx = ((a_idx + 1) % self.m)
        if 0 == a_idx:
            b_idx = ((b_idx + 1) % self.n)

        return a_idx, b_idx, c_idx

class mm_rsg_bfirst(mm_mxn_register_sequence_generator):
    def advance_simplified(self, a_idx : int, b_idx : int, c_idx : int) -> tuple[int,int,int]:

        c_idx = ((c_idx + 1) % (self.m*self.n))
        if 0 == c_idx:
            self.a_idx_offset += self.m
            self.a_idx_offset = self.a_idx_offset % len(self.a_data_register_list)
            self.b_idx_offset += self.n
            self.b_idx_offset = self.b_idx_offset % len(self.b_data_register_list)
        b_idx = ((b_idx + 1) % self.n)
        if 0 == b_idx:
            a_idx = ((a_idx + 1) % self.m)

        return a_idx, b_idx, c_idx
