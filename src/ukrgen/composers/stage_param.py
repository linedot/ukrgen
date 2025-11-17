from dataclasses import dataclass

@dataclass
class stage_param:
    value       : str|list[str]
    description : str
    default     : str|list[str]|None = None
    choices     : list[str]|None = None
    required    : bool = True
