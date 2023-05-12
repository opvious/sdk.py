import dataclasses
import math

from ..common import encode_extended_float


@dataclasses.dataclass(frozen=True)
class Image:
    """A tensor's set of possible values

    Various convenience factory methods are provided for common cases:

    * :func:`indicator` for indicator values
    * :func:`non_negative`, :func:`non_positive`, :func:`unit` for continuous
      values
    * :func:`natural`, :func:`integral` for discrete values
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


def indicator() -> Image:
    """Returns an image representing `{0, 1}` tensors"""
    return Image(lower_bound=0, upper_bound=1, is_integral=True)


def non_negative(upper_bound=math.inf) -> Image:
    """Returns an image representing non-negative reals

    Args:
        upper_bound: Optional inclusive upper bound
    """
    return Image(lower_bound=0, upper_bound=upper_bound)


def non_positive(lower_bound=-math.inf) -> Image:
    """Returns an image representing non-positive reals

    Args:
        lower_bound: Optional inclusive lower bound
    """
    return Image(lower_bound=lower_bound, upper_bound=0)


def unit() -> Image:
    """Returns an image representing the `[0, 1]` segment"""
    return Image(lower_bound=0, upper_bound=1)


def natural(upper_bound=math.inf) -> Image:
    """Returns an image representing natural (non-negative) integers

    Args:
        upper_bound: Optional inclusive upper bound
    """
    return Image(lower_bound=0, upper_bound=upper_bound, is_integral=True)


def integral() -> Image:
    """Returns an image representing all integers"""
    return Image(is_integral=True)
