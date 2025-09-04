# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import math
from functools import reduce

from omegaconf import OmegaConf

# fmt: off
OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
OmegaConf.register_new_resolver("subtract", lambda *numbers: reduce(lambda x, y: x - y, numbers))
OmegaConf.register_new_resolver("mul", lambda *numbers: reduce(lambda x, y: x * y, numbers))
OmegaConf.register_new_resolver("mul_int", lambda *numbers: int(reduce(lambda x, y: x * y, numbers)))
OmegaConf.register_new_resolver("div_int", lambda *numbers: int(reduce(lambda x, y: x / y, numbers)))
OmegaConf.register_new_resolver("div_float", lambda *numbers: reduce(lambda x, y: x / y, numbers))
OmegaConf.register_new_resolver("sqrt_div", lambda x, y: math.sqrt(x / y))
# fmt: on
