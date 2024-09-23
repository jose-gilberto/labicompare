from .utils import calculate_average_ranks
from ._nemenyi import nemenyi
from ._bonferroni import bonferroni_dunn
from ._wilcoxon import wilcoxon_holm

__all__ = [
    'calculate_average_ranks',
    'nemenyi',
    'bonferroni_dunn',
    'wilcoxon_holm',
]