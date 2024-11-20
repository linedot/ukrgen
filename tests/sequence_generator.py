import unittest

from algobuild.sequence_generator import sequence_generator


class sequence_generator_tester(sequence_generator):
    required_parameters : list[str] = ["start_value","step","rollover"]
    side_effects : list[str] = ["on_rollover"]

    def initialize(self):
        self.value = self.internal_parameters["start_value"]
        self.step = self.internal_parameters["step"]
        self.rollover = self.internal_parameters["rollover"]

        assert (self.value < self.rollover), "specified value higher than rollover"

        self.on_rollover = self.side_effect_handlers["on_rollover"]

    def advance(self):
        current_values = {
            "value" : self.value
        }

        self.value += self.step
        if self.rollover <= self.value:
            self.on_rollover(self.value)
            self.value = 0

        return current_values

class sequence_generator_testcase(unittest.TestCase):
    
    def test_missing_side_effect_handler(self):
        parameters = {
            "start_value" : 3,
            "step" : 1,
            "rollover" : 10
        }
        side_effect_handlers = {
        }
        with self.assertRaises(ValueError) as context:
            sgt = sequence_generator_tester(parameters=parameters, 
                                            side_effect_handlers=side_effect_handlers)
        self.assertTrue("No handler for side effect on_rollover specified" in str(context.exception))
    
    def test_missing_parameter(self):
        rollovers = []
        parameters = {
            "start_value" : 3,
            "rollover" : 10
        }
        side_effect_handlers = {
            "on_rollover" : lambda i : rollovers.append(i)
        }
        with self.assertRaises(ValueError) as context:
            sgt = sequence_generator_tester(parameters=parameters, 
                                            side_effect_handlers=side_effect_handlers)
        self.assertTrue("required parameter step not in parameters" in str(context.exception))

    def test_rollover(self):
        rollovers = []
        parameters = {
            "start_value" : 3,
            "step" : 1,
            "rollover" : 10
        }
        side_effect_handlers = {
            "on_rollover" : lambda i : rollovers.append(i)
        }
        sgt = sequence_generator_tester(parameters=parameters, 
                                        side_effect_handlers=side_effect_handlers)


        sequence = [sgt.advance() for i in range(10)] 
        
        self.assertEqual(sequence,[
            {"value": 3},
            {"value": 4},
            {"value": 5},
            {"value": 6},
            {"value": 7},
            {"value": 8},
            {"value": 9},
            {"value": 0},
            {"value": 1},
            {"value": 2},
        ])
        self.assertEqual(rollovers,[10])
