import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class Image:
    lower_bound: float = -math.inf
    upper_bound: float = math.inf
    is_integral: bool = False


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
