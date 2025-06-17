import unittest

from ukrgen.compute.trsm import trsm_l_rsg_topdown,trsm_u_rsg_topdown


class test_trsm_rsg(unittest.TestCase):
    def test_l_4x2v_sequence(self):
        parameters = {
            "a_data_register_list" : [i   for i in range(10)],
            "b_data_register_list" : [i   for i in range(8)],

            "a_data_register_list_start" : 0,
            "b_data_register_list_start" : 0,
            "m" : 4,
            "n" : 2
        }
        side_effect_handlers = {
        }

        sg = trsm_l_rsg_topdown(parameters=parameters, 
                           side_effect_handlers=side_effect_handlers)

        sequence = [str(sg.advance()) for i in range(20)]

        print("\n".join(sequence))


    def test_u_4x2v_sequence(self):
        parameters = {
            "a_data_register_list" : [i   for i in range(10)],
            "b_data_register_list" : [i   for i in range(8)],

            "a_data_register_list_start" : 0,
            "b_data_register_list_start" : 0,
            "m" : 4,
            "n" : 2
        }
        side_effect_handlers = {
        }

        sg = trsm_u_rsg_topdown(parameters=parameters, 
                           side_effect_handlers=side_effect_handlers)

        sequence = [str(sg.advance()) for i in range(20)]

        print("\n".join(sequence))


    def test_u_4x3v_sequence(self):
        parameters = {
            "a_data_register_list" : [i   for i in range(8)],
            "b_data_register_list" : [i   for i in range(24)],

            "a_data_register_list_start" : 0,
            "b_data_register_list_start" : 0,
            "m" : 8,
            "n" : 3
        }
        side_effect_handlers = {
        }

        sg = trsm_u_rsg_topdown(parameters=parameters, 
                           side_effect_handlers=side_effect_handlers)

        sequence = [str(sg.advance()) for i in range(108)]

        print("\n".join(sequence))
