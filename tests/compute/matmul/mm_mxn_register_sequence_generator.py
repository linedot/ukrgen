import unittest

from algobuild.compute.matmul import mm_rsg_afirst,mm_rsg_bfirst,mm_rsg_bfirst_reorderc


class test_mm_rsg(unittest.TestCase):
    def test_2vx4_sequence_afirst(self):
        parameters = {
            "a_data_register_list" : [i   for i in range(4)],
            "b_data_register_list" : [i   for i in range(8)],
            "c_data_register_list" : [i+4 for i in range(8)],

            "a_data_register_list_start" : 0,
            "b_data_register_list_start" : 0,
            "c_data_register_list_start" : 0,
            "m" : 2,
            "n" : 4
        }
        side_effect_handlers = {
        }

        sg = mm_rsg_afirst(parameters=parameters, 
                           side_effect_handlers=side_effect_handlers)

        sequence = [sg.advance() for i in range(16)]

        expected_sequence = [
            {'a_vector': 0, 'b_vector': 0, 'c_vector': 4 , 'a_data_offset': 0, 'b_data_offset': 0, 'c_data_offset': 0},
            {'a_vector': 1, 'b_vector': 0, 'c_vector': 5 , 'a_data_offset': 1, 'b_data_offset': 0, 'c_data_offset': 1},
            {'a_vector': 0, 'b_vector': 1, 'c_vector': 6 , 'a_data_offset': 0, 'b_data_offset': 1, 'c_data_offset': 2},
            {'a_vector': 1, 'b_vector': 1, 'c_vector': 7 , 'a_data_offset': 1, 'b_data_offset': 1, 'c_data_offset': 3},
            {'a_vector': 0, 'b_vector': 2, 'c_vector': 8 , 'a_data_offset': 0, 'b_data_offset': 2, 'c_data_offset': 4},
            {'a_vector': 1, 'b_vector': 2, 'c_vector': 9 , 'a_data_offset': 1, 'b_data_offset': 2, 'c_data_offset': 5},
            {'a_vector': 0, 'b_vector': 3, 'c_vector': 10, 'a_data_offset': 0, 'b_data_offset': 3, 'c_data_offset': 6},
            {'a_vector': 1, 'b_vector': 3, 'c_vector': 11, 'a_data_offset': 1, 'b_data_offset': 3, 'c_data_offset': 7},
            {'a_vector': 2, 'b_vector': 4, 'c_vector': 4 , 'a_data_offset': 2, 'b_data_offset': 4, 'c_data_offset': 0},
            {'a_vector': 3, 'b_vector': 4, 'c_vector': 5 , 'a_data_offset': 3, 'b_data_offset': 4, 'c_data_offset': 1},
            {'a_vector': 2, 'b_vector': 5, 'c_vector': 6 , 'a_data_offset': 2, 'b_data_offset': 5, 'c_data_offset': 2},
            {'a_vector': 3, 'b_vector': 5, 'c_vector': 7 , 'a_data_offset': 3, 'b_data_offset': 5, 'c_data_offset': 3},
            {'a_vector': 2, 'b_vector': 6, 'c_vector': 8 , 'a_data_offset': 2, 'b_data_offset': 6, 'c_data_offset': 4},
            {'a_vector': 3, 'b_vector': 6, 'c_vector': 9 , 'a_data_offset': 3, 'b_data_offset': 6, 'c_data_offset': 5},
            {'a_vector': 2, 'b_vector': 7, 'c_vector': 10, 'a_data_offset': 2, 'b_data_offset': 7, 'c_data_offset': 6},
            {'a_vector': 3, 'b_vector': 7, 'c_vector': 11, 'a_data_offset': 3, 'b_data_offset': 7, 'c_data_offset': 7}
        ]

        self.assertEqual(sequence, expected_sequence)

    def test_2vx4_2ar4br8cr_sequence_afirst(self):
        parameters = {
            "a_data_register_list" : [i   for i in range(2)],
            "b_data_register_list" : [i   for i in range(4)],
            "c_data_register_list" : [i+2 for i in range(8)],

            "a_data_register_list_start" : 0,
            "b_data_register_list_start" : 0,
            "c_data_register_list_start" : 0,
            "m" : 2,
            "n" : 4
        }
        side_effect_handlers = {
        }

        sg = mm_rsg_afirst(parameters=parameters, 
                           side_effect_handlers=side_effect_handlers)

        sequence = [sg.advance() for i in range(16)]

        expected_sequence = [
            {'a_vector': 0, 'b_vector': 0, 'c_vector': 2, 'a_data_offset': 0, 'b_data_offset': 0, 'c_data_offset': 0},
            {'a_vector': 1, 'b_vector': 0, 'c_vector': 3, 'a_data_offset': 1, 'b_data_offset': 0, 'c_data_offset': 1},
            {'a_vector': 0, 'b_vector': 1, 'c_vector': 4, 'a_data_offset': 0, 'b_data_offset': 1, 'c_data_offset': 2},
            {'a_vector': 1, 'b_vector': 1, 'c_vector': 5, 'a_data_offset': 1, 'b_data_offset': 1, 'c_data_offset': 3},
            {'a_vector': 0, 'b_vector': 2, 'c_vector': 6, 'a_data_offset': 0, 'b_data_offset': 2, 'c_data_offset': 4},
            {'a_vector': 1, 'b_vector': 2, 'c_vector': 7, 'a_data_offset': 1, 'b_data_offset': 2, 'c_data_offset': 5},
            {'a_vector': 0, 'b_vector': 3, 'c_vector': 8, 'a_data_offset': 0, 'b_data_offset': 3, 'c_data_offset': 6},
            {'a_vector': 1, 'b_vector': 3, 'c_vector': 9, 'a_data_offset': 1, 'b_data_offset': 3, 'c_data_offset': 7},
            {'a_vector': 0, 'b_vector': 0, 'c_vector': 2, 'a_data_offset': 2, 'b_data_offset': 4, 'c_data_offset': 0},
            {'a_vector': 1, 'b_vector': 0, 'c_vector': 3, 'a_data_offset': 3, 'b_data_offset': 4, 'c_data_offset': 1},
            {'a_vector': 0, 'b_vector': 1, 'c_vector': 4, 'a_data_offset': 2, 'b_data_offset': 5, 'c_data_offset': 2},
            {'a_vector': 1, 'b_vector': 1, 'c_vector': 5, 'a_data_offset': 3, 'b_data_offset': 5, 'c_data_offset': 3},
            {'a_vector': 0, 'b_vector': 2, 'c_vector': 6, 'a_data_offset': 2, 'b_data_offset': 6, 'c_data_offset': 4},
            {'a_vector': 1, 'b_vector': 2, 'c_vector': 7, 'a_data_offset': 3, 'b_data_offset': 6, 'c_data_offset': 5},
            {'a_vector': 0, 'b_vector': 3, 'c_vector': 8, 'a_data_offset': 2, 'b_data_offset': 7, 'c_data_offset': 6},
            {'a_vector': 1, 'b_vector': 3, 'c_vector': 9, 'a_data_offset': 3, 'b_data_offset': 7, 'c_data_offset': 7}
        ]

        self.assertEqual(sequence, expected_sequence)

    def test_2vx4_sequence_bfirst(self):
        m = 2
        n = 4
        acount = 4
        bcount = 8
        ccount = 8
    
        # NOTE: this is expecting the data to be as arranged as with the afirst
        #       version. Possibly the "shuffling" could be performed in the rsg
        #       or no shuffling performed at all depending on how it's used
        # TODO: explore more cases and develop good argument for the
        #       decision where this "shuffling/rearrangement" should be done
        # NOTE: Now that we're generating data offsets as well, the drawback
        #       of this approach is clear: the data offsets aren't shuffled
        # c_registers = [(i*m)%ccount+(i*m)//ccount for i in range(ccount)]
        # c_registers = [i+acount for i in c_registers]
        c_registers = [i+acount for i in range(ccount)]

        parameters = {
            "a_data_register_list" : [i   for i in range(acount)],
            "b_data_register_list" : [i   for i in range(bcount)],
            "c_data_register_list" : c_registers,

            "a_data_register_list_start" : 0,
            "b_data_register_list_start" : 0,
            "c_data_register_list_start" : 0,
            "m" : m,
            "n" : n
        }
        side_effect_handlers = {
        }

        sg = mm_rsg_bfirst(parameters=parameters, 
                           side_effect_handlers=side_effect_handlers)

        sequence = [sg.advance() for i in range(16)]

        expected_sequence = [
            {'a_vector': 0, 'b_vector': 0, 'c_vector': 4 , 'a_data_offset': 0, 'b_data_offset': 0, 'c_data_offset': 0},
            {'a_vector': 0, 'b_vector': 1, 'c_vector': 5 , 'a_data_offset': 0, 'b_data_offset': 1, 'c_data_offset': 1},
            {'a_vector': 0, 'b_vector': 2, 'c_vector': 6 , 'a_data_offset': 0, 'b_data_offset': 2, 'c_data_offset': 2},
            {'a_vector': 0, 'b_vector': 3, 'c_vector': 7 , 'a_data_offset': 0, 'b_data_offset': 3, 'c_data_offset': 3},
            {'a_vector': 1, 'b_vector': 0, 'c_vector': 8 , 'a_data_offset': 1, 'b_data_offset': 0, 'c_data_offset': 4},
            {'a_vector': 1, 'b_vector': 1, 'c_vector': 9 , 'a_data_offset': 1, 'b_data_offset': 1, 'c_data_offset': 5},
            {'a_vector': 1, 'b_vector': 2, 'c_vector': 10, 'a_data_offset': 1, 'b_data_offset': 2, 'c_data_offset': 6},
            {'a_vector': 1, 'b_vector': 3, 'c_vector': 11, 'a_data_offset': 1, 'b_data_offset': 3, 'c_data_offset': 7},
            {'a_vector': 2, 'b_vector': 4, 'c_vector': 4 , 'a_data_offset': 2, 'b_data_offset': 4, 'c_data_offset': 0},
            {'a_vector': 2, 'b_vector': 5, 'c_vector': 5 , 'a_data_offset': 2, 'b_data_offset': 5, 'c_data_offset': 1},
            {'a_vector': 2, 'b_vector': 6, 'c_vector': 6 , 'a_data_offset': 2, 'b_data_offset': 6, 'c_data_offset': 2},
            {'a_vector': 2, 'b_vector': 7, 'c_vector': 7 , 'a_data_offset': 2, 'b_data_offset': 7, 'c_data_offset': 3},
            {'a_vector': 3, 'b_vector': 4, 'c_vector': 8 , 'a_data_offset': 3, 'b_data_offset': 4, 'c_data_offset': 4},
            {'a_vector': 3, 'b_vector': 5, 'c_vector': 9 , 'a_data_offset': 3, 'b_data_offset': 5, 'c_data_offset': 5},
            {'a_vector': 3, 'b_vector': 6, 'c_vector': 10, 'a_data_offset': 3, 'b_data_offset': 6, 'c_data_offset': 6},
            {'a_vector': 3, 'b_vector': 7, 'c_vector': 11, 'a_data_offset': 3, 'b_data_offset': 7, 'c_data_offset': 7} 
        ]

        self.assertEqual(sequence, expected_sequence)


    def test_2vx4_sequence_bfirst_reorderc(self):
        m = 2
        n = 4
        acount = 4
        bcount = 8
        ccount = 8
    
        c_registers = [i+acount for i in range(ccount)]

        parameters = {
            "a_data_register_list" : [i   for i in range(acount)],
            "b_data_register_list" : [i   for i in range(bcount)],
            "c_data_register_list" : c_registers,

            "a_data_register_list_start" : 0,
            "b_data_register_list_start" : 0,
            "c_data_register_list_start" : 0,
            "m" : m,
            "n" : n
        }
        side_effect_handlers = {
        }

        sg = mm_rsg_bfirst_reorderc(parameters=parameters, 
                           side_effect_handlers=side_effect_handlers)

        sequence = [sg.advance() for i in range(16)]

        print("\n".join([str(x) for x in sequence]))

        expected_sequence = [
            {'a_vector': 0, 'b_vector': 0, 'c_vector': 4 , 'a_data_offset': 0, 'b_data_offset': 0, 'c_data_offset': 0},
            {'a_vector': 0, 'b_vector': 1, 'c_vector': 6 , 'a_data_offset': 0, 'b_data_offset': 1, 'c_data_offset': 2},
            {'a_vector': 0, 'b_vector': 2, 'c_vector': 8 , 'a_data_offset': 0, 'b_data_offset': 2, 'c_data_offset': 4},
            {'a_vector': 0, 'b_vector': 3, 'c_vector': 10, 'a_data_offset': 0, 'b_data_offset': 3, 'c_data_offset': 6},
            {'a_vector': 1, 'b_vector': 0, 'c_vector': 5 , 'a_data_offset': 1, 'b_data_offset': 0, 'c_data_offset': 1},
            {'a_vector': 1, 'b_vector': 1, 'c_vector': 7 , 'a_data_offset': 1, 'b_data_offset': 1, 'c_data_offset': 3},
            {'a_vector': 1, 'b_vector': 2, 'c_vector': 9 , 'a_data_offset': 1, 'b_data_offset': 2, 'c_data_offset': 5},
            {'a_vector': 1, 'b_vector': 3, 'c_vector': 11, 'a_data_offset': 1, 'b_data_offset': 3, 'c_data_offset': 7},
            {'a_vector': 2, 'b_vector': 4, 'c_vector': 4 , 'a_data_offset': 2, 'b_data_offset': 4, 'c_data_offset': 0},
            {'a_vector': 2, 'b_vector': 5, 'c_vector': 6 , 'a_data_offset': 2, 'b_data_offset': 5, 'c_data_offset': 2},
            {'a_vector': 2, 'b_vector': 6, 'c_vector': 8 , 'a_data_offset': 2, 'b_data_offset': 6, 'c_data_offset': 4},
            {'a_vector': 2, 'b_vector': 7, 'c_vector': 10, 'a_data_offset': 2, 'b_data_offset': 7, 'c_data_offset': 6},
            {'a_vector': 3, 'b_vector': 4, 'c_vector': 5 , 'a_data_offset': 3, 'b_data_offset': 4, 'c_data_offset': 1},
            {'a_vector': 3, 'b_vector': 5, 'c_vector': 7 , 'a_data_offset': 3, 'b_data_offset': 5, 'c_data_offset': 3},
            {'a_vector': 3, 'b_vector': 6, 'c_vector': 9 , 'a_data_offset': 3, 'b_data_offset': 6, 'c_data_offset': 5},
            {'a_vector': 3, 'b_vector': 7, 'c_vector': 11, 'a_data_offset': 3, 'b_data_offset': 7, 'c_data_offset': 7} 
        ]

        self.assertEqual(sequence, expected_sequence)
