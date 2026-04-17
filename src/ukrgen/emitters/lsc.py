from ..models.load_store_operations import (
        lsc_add_val_off,
        lsc_addr_add,
        lsc_debugmsg,
        lsc_load,
        lsc_store,
        lsc_transformation,
        lsc_zero,
        )

from ..models.loop import lsc_loop

from ..specializers.special_treg_ldst import (
        lsc_treg_row_load,
        lsc_treg_row_store
        )


from asmgen.registers import asm_data_type as adt

from abc import ABC, abstractmethod
from functools import singledispatchmethod


class lsc_emitter(ABC):

    REQUIRED_LSC_OPERATIONS={
            lsc_add_val_off,
            lsc_addr_add,
            lsc_debugmsg,
            lsc_load,
            lsc_store,
            lsc_transformation,
            lsc_zero,
            lsc_treg_row_load,
            lsc_treg_row_store,
            lsc_loop
            }

    def transform_list(self, ops : list[lsc_operation],
                component_dts : dict[str,adt]) -> list[str]:
        result = []
        for op in ops:
            transformed_op = self.transform(op, component_dts)
            result.append(transformed_op)
        return result


    @singledispatchmethod
    @abstractmethod
    def transform(self, op) -> str:
        """
        Outputs string containing IR/ASM/... corresponding to the LSC operation 
        """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        transform_method = cls.__dict__.get("transform")

        if transform_method is None:
            raise TypeError(
                f"'{cls.__name__}' must override 'transform' with @singledispatchmethod "
                f"to create its own registry."
            )

        if not isinstance(transform_method, singledispatchmethod):
            raise TypeError(
                f"'{cls.__name__}.transform' must be decorated with @singledispatchmethod."
            )

        # 4. Now it is safe to access the dispatcher and registry
        registered_types = set(transform_method.dispatcher.registry.keys())
        missing_ops = cls.REQUIRED_LSC_OPERATIONS - registered_types

        if missing_ops:
            missing_names = [op.__name__ for op in missing_ops]
            raise TypeError(
                f"Implementation incomplete! '{cls.__name__}' is missing "
                f"@transform.register handlers for: {', '.join(missing_names)}"
            )
