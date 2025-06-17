# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import abc

from typing import Callable

class sequence_generator:
    required_parameters : list[str]
    side_effects : list[str]

    def __init__(self,
                 parameters : dict[str,int|list[int]],
                 side_effect_handlers : dict[str,Callable[dict[str,int|list[int]],None]]):
        for param_name in self.required_parameters:
            if param_name not in parameters:
                raise ValueError(f"required parameter {param_name} not in parameters")
        self.internal_parameters = {k:v for k,v
                                    in parameters.items()
                                    if k in self.required_parameters}

        for side_effect_name in self.side_effects:
            if side_effect_name not in side_effect_handlers:
                raise ValueError(f"No handler for side effect {side_effect_name} specified")
        self.side_effect_handlers = {k:v for k,v
                                     in side_effect_handlers.items()
                                     if k in self.side_effects}
        self.initialize()

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError("Abstract method called")

    @abc.abstractmethod
    def advance(self) -> dict[str,int]:
        raise NotImplementedError("Abstract method called")
