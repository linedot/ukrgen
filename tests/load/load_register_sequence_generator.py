import unittest

from algobuild.load import lrsg_simple

class test_lrsg_simple(unittest.TestCase):
    def test_non_unique_address_registers(self):
        rollovers = []
        parameters = {
            "address_register_list" : [0, 0],
            "address_register_list_start" : 0,
            "offset_start_list" : [0, 0],
            "offset_threshold_list" : [1, 1],
            "offset_reset_value_list" : [0, 0],
            "offset_step" : 1,
            "max_offset" : 1,
            "data_register_list" : [0,2,4,6],
            "data_register_list_start" : 0
        }
        side_effect_handlers = {
            "on_offset_rollover" : lambda reg, off: rollovers.append((reg,off))
        }

        with self.assertRaises(ValueError) as context:
            sg = lrsg_simple(parameters=parameters, 
                             side_effect_handlers=side_effect_handlers)
        self.assertTrue("non-unique-entries in address register list" in str(context.exception))

    def test_non_unique_data_registers(self):
        rollovers = []
        parameters = {
            "address_register_list" : [0, 1],
            "address_register_list_start" : 0,
            "offset_start_list" : [0, 0],
            "offset_threshold_list" : [1, 1],
            "offset_reset_value_list" : [0, 0],
            "offset_step" : 1,
            "max_offset" : 1,
            "data_register_list" : [0,2,2,6],
            "data_register_list_start" : 0
        }
        side_effect_handlers = {
            "on_offset_rollover" : lambda reg, off: rollovers.append((reg,off))
        }

        with self.assertRaises(ValueError) as context:
            sg = lrsg_simple(parameters=parameters, 
                             side_effect_handlers=side_effect_handlers)
        self.assertTrue("non-unique-entries in data register list" in str(context.exception))

    def test_offsets_start_above_threshold(self):
        rollovers = []
        parameters = {
            "address_register_list" : [0, 1],
            "address_register_list_start" : 0,
            "offset_start_list" : [5, 5],
            "offset_threshold_list" : [1, 1],
            "offset_reset_value_list" : [0, 0],
            "offset_step" : 1,
            "max_offset" : 5,
            "data_register_list" : [0,2,2,6],
            "data_register_list_start" : 0
        }
        side_effect_handlers = {
            "on_offset_rollover" : lambda reg, off: rollovers.append((reg,off))
        }

        with self.assertRaises(ValueError) as context:
            sg = lrsg_simple(parameters=parameters, 
                             side_effect_handlers=side_effect_handlers)
        self.assertTrue("starting offset (5) for reg 0 higher than it's threshold (1)" in str(context.exception))


    def test_offsets_start_above_max_offset(self):
        rollovers = []
        parameters = {
            "address_register_list" : [0, 1],
            "address_register_list_start" : 0,
            "offset_start_list" : [5, 5],
            "offset_threshold_list" : [5, 5],
            "offset_reset_value_list" : [0, 0],
            "offset_step" : 1,
            "max_offset" : 1,
            "data_register_list" : [0,2,2,6],
            "data_register_list_start" : 0
        }
        side_effect_handlers = {
            "on_offset_rollover" : lambda reg, off: rollovers.append((reg,off))
        }

        with self.assertRaises(ValueError) as context:
            sg = lrsg_simple(parameters=parameters, 
                             side_effect_handlers=side_effect_handlers)
        self.assertTrue("starting offset (5) for reg 0 higher than max offset (1)" in str(context.exception))

    def test_1areg_6dreg_simple(self):
        rollovers = []
        parameters = {
            "address_register_list" : [0],
            "address_register_list_start" : 0,
            "offset_start_list" : [0],
            "offset_threshold_list" : [15],
            "offset_reset_value_list" : [0, 0],
            "offset_step" : 8,
            "max_offset" : 15,
            "data_register_list" : [0,2,4,6,8,10],
            "data_register_list_start" : 0
        }
        side_effect_handlers = {
            "on_offset_rollover" : lambda address_register, offset: rollovers.append((address_register,offset))
        }

        sg = lrsg_simple(parameters=parameters, 
                         side_effect_handlers=side_effect_handlers)

        sequence = [sg.advance() for i in range(8)]

        self.assertEqual(sequence,[
            {'address_register': 0, 'data_register': 0, 'offset': 0},
            {'address_register': 0, 'data_register': 2, 'offset': 8},
            {'address_register': 0, 'data_register': 4, 'offset': 0},
            {'address_register': 0, 'data_register': 6, 'offset': 8},
            {'address_register': 0, 'data_register': 8, 'offset': 0},
            {'address_register': 0, 'data_register': 10, 'offset': 8},
            {'address_register': 0, 'data_register': 0, 'offset': 0},
            {'address_register': 0, 'data_register': 2, 'offset': 8}
        ])
        self.assertEqual(rollovers,[
            (0,16),
            (0,16),
            (0,16),
            (0,16),
        ])

    def test_2areg_12dreg_simple(self):
        rollovers = []
        parameters = {
            "address_register_list" : [0,1],
            "address_register_list_start" : 0,
            "offset_start_list" : [0,0],
            "offset_threshold_list" : [31,31],
            "offset_reset_value_list" : [0, 0],
            "offset_step" : 8,
            "max_offset" : 31,
            "data_register_list" : [i for i in range(12)],
            "data_register_list_start" : 0
        }
        side_effect_handlers = {
            "on_offset_rollover" : lambda address_register, offset: rollovers.append((address_register,offset))
        }

        sg = lrsg_simple(parameters=parameters, 
                         side_effect_handlers=side_effect_handlers)

        sequence = [sg.advance() for i in range(16)]

        self.assertEqual(sequence,[
            {'address_register': 0, 'data_register': 0, 'offset': 0},
            {'address_register': 0, 'data_register': 1, 'offset': 8},
            {'address_register': 0, 'data_register': 2, 'offset': 16},
            {'address_register': 0, 'data_register': 3, 'offset': 24},
            {'address_register': 1, 'data_register': 4, 'offset': 0},
            {'address_register': 1, 'data_register': 5, 'offset': 8},
            {'address_register': 1, 'data_register': 6, 'offset': 16},
            {'address_register': 1, 'data_register': 7, 'offset': 24},
            {'address_register': 0, 'data_register': 8, 'offset': 0},
            {'address_register': 0, 'data_register': 9, 'offset': 8},
            {'address_register': 0, 'data_register': 10, 'offset': 16},
            {'address_register': 0, 'data_register': 11, 'offset': 24},
            {'address_register': 1, 'data_register': 0, 'offset': 0},
            {'address_register': 1, 'data_register': 1, 'offset': 8},
            {'address_register': 1, 'data_register': 2, 'offset': 16},
            {'address_register': 1, 'data_register': 3, 'offset': 24},
        ])
        self.assertEqual(rollovers,[
            (0,32),
            (1,32),
            (0,32),
            (1,32),
        ])

    def test_2areg_12dreg_unequal_thresholds(self):
        rollovers = []
        parameters = {
            "address_register_list" : [0,1],
            "address_register_list_start" : 0,
            "offset_start_list" : [0,0],
            "offset_threshold_list" : [31,15],
            "offset_reset_value_list" : [0, 0],
            "offset_step" : 8,
            "max_offset" : 31,
            "data_register_list" : [i for i in range(12)],
            "data_register_list_start" : 0
        }
        side_effect_handlers = {
            "on_offset_rollover" : lambda address_register, offset: rollovers.append((address_register,offset))
        }

        sg = lrsg_simple(parameters=parameters, 
                         side_effect_handlers=side_effect_handlers)

        sequence = [sg.advance() for i in range(16)]

        self.assertEqual(sequence,[
            {'address_register': 0, 'data_register': 0, 'offset': 0},
            {'address_register': 0, 'data_register': 1, 'offset': 8},
            {'address_register': 0, 'data_register': 2, 'offset': 16},
            {'address_register': 0, 'data_register': 3, 'offset': 24},
            {'address_register': 1, 'data_register': 4, 'offset': 0},
            {'address_register': 1, 'data_register': 5, 'offset': 8},
            {'address_register': 0, 'data_register': 6, 'offset': 0},
            {'address_register': 0, 'data_register': 7, 'offset': 8},
            {'address_register': 0, 'data_register': 8, 'offset': 16},
            {'address_register': 0, 'data_register': 9, 'offset': 24},
            {'address_register': 1, 'data_register': 10, 'offset': 0},
            {'address_register': 1, 'data_register': 11, 'offset': 8},
            {'address_register': 0, 'data_register': 0, 'offset': 0},
            {'address_register': 0, 'data_register': 1, 'offset': 8},
            {'address_register': 0, 'data_register': 2, 'offset': 16},
            {'address_register': 0, 'data_register': 3, 'offset': 24},
        ])
        self.assertEqual(rollovers,[
            (0,32),
            (1,16),
            (0,32),
            (1,16),
            (0,32),
        ])

    def test_2areg_12dreg_negative_start(self):
        rollovers = []
        parameters = {
            "address_register_list" : [0,1],
            "address_register_list_start" : 0,
            "offset_start_list" : [-16,-16],
            "offset_threshold_list" : [15,15],
            "offset_reset_value_list" : [-16, -16],
            "offset_step" : 8,
            "max_offset" : 31,
            "data_register_list" : [i for i in range(12)],
            "data_register_list_start" : 0
        }
        side_effect_handlers = {
            "on_offset_rollover" : lambda address_register, offset: rollovers.append((address_register,offset))
        }

        sg = lrsg_simple(parameters=parameters, 
                         side_effect_handlers=side_effect_handlers)

        sequence = [sg.advance() for i in range(16)]

        self.assertEqual(sequence,[
            {'address_register': 0, 'data_register': 0, 'offset': -16},
            {'address_register': 0, 'data_register': 1, 'offset': -8},
            {'address_register': 0, 'data_register': 2, 'offset': 0},
            {'address_register': 0, 'data_register': 3, 'offset': 8},
            {'address_register': 1, 'data_register': 4, 'offset': -16},
            {'address_register': 1, 'data_register': 5, 'offset': -8},
            {'address_register': 1, 'data_register': 6, 'offset': 0},
            {'address_register': 1, 'data_register': 7, 'offset': 8},
            {'address_register': 0, 'data_register': 8, 'offset': -16},
            {'address_register': 0, 'data_register': 9, 'offset': -8},
            {'address_register': 0, 'data_register': 10, 'offset': 0},
            {'address_register': 0, 'data_register': 11, 'offset': 8},
            {'address_register': 1, 'data_register': 0, 'offset': -16},
            {'address_register': 1, 'data_register': 1, 'offset': -8},
            {'address_register': 1, 'data_register': 2, 'offset': 0},
            {'address_register': 1, 'data_register': 3, 'offset': 8},
        ])
        self.assertEqual(rollovers,[
            (0,32),
            (1,32),
            (0,32),
            (1,32),
        ])
