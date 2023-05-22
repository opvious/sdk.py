from __future__ import annotations

import dataclasses
import math

from ...common import encode_extended_float


@dataclasses.dataclass(frozen=True)
class Image:
    """A tensor's set of possible values

    See the methods below various convenience factories.
    """

    lower_bound: float = -math.inf
    """The image's smallest value (inclusive)"""

    upper_bound: float = math.inf
    """The image's largest value (inclusive)"""

    is_integral: bool = False
    """Whether the image only contain integers"""

    def render(self) -> str:
        if self.upper_bound == math.inf:
            if self.lower_bound == -math.inf:
                return "\\mathbb{Z}" if self.is_integral else "\\mathbb{R}"
            if self.lower_bound == 0:
                return "\\mathbb{N}" if self.is_integral else "\\mathbb{R}_+"
        lb = encode_extended_float(self.lower_bound)
        ub = encode_extended_float(self.upper_bound)
        if self.is_integral:
            if lb == 0 and ub == 1:
                return "\\{0, 1\\}"
            return f"\\{{{lb} \\ldots {ub}\\}}"
        return f"[{lb}, {ub}]"
