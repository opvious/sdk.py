import dataclasses
import math

from ..common import encode_extended_float


@dataclasses.dataclass(frozen=True)
class Image:
    lower_bound: float = -math.inf
    upper_bound: float = math.inf
    is_integral: bool = False

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
                return "{0, 1}"
            return f"{{{lb} \\ldots {ub}}}"
        return f"[{lb}, {ub}]"


def indicator() -> Image:
    return Image(lower_bound=0, upper_bound=1, is_integral=True)


def non_negative_real(upper_bound=math.inf) -> Image:
    return Image(lower_bound=0, upper_bound=upper_bound)


def non_positive_real(lower_bound=-math.inf) -> Image:
    return Image(lower_bound=lower_bound, upper_bound=0)


def unit() -> Image:
    return Image(lower_bound=0, upper_bound=1)


def natural(upper_bound=math.inf) -> Image:
    return Image(lower_bound=0, upper_bound=upper_bound, is_integral=True)


def integral() -> Image:
    return Image(is_integral=True)
