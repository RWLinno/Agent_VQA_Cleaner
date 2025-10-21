"""
Utils module - 工具函数
"""
from .Heuristic_Rules_Internvl2_5 import (
    is_super_long_sentence,
    calculate_ngram_repetition,
    check_conversations_repetition,
    flag_function_1,
    flag_function_2,
    flag_function_3,
    flag_function_4
)

__all__ = [
    'is_super_long_sentence',
    'calculate_ngram_repetition',
    'check_conversations_repetition',
    'flag_function_1',
    'flag_function_2',
    'flag_function_3',
    'flag_function_4'
]

