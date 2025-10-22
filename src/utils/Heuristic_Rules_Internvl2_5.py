import re
from collections import Counter

def is_super_long_sentence(text: str, n: int) -> bool:
    """
    Check if the text contains any sentence with at least `n` words
    (counting only words longer than 5 characters).
    
    Returns True if a super long sentence is found.
    """
    sentences = re.split(r'[,.!?\n]+', text)
    for sentence in sentences:
        words = sentence.strip().split()
        words = [item.strip() for item in words if len(item.strip()) > 5]
        if len(words) >= n:
            return True
    return False


def calculate_ngram_repetition(text: str, n: int) -> float:
    """
    Calculate the n-gram repetition ratio in a text.
    
    Returns the fraction of n-grams that appear more than once.
    """
    words = text.split()
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
    return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0


def check_unique_word_ratio(text: str, min_ratio: float = 0.3) -> bool:
    """
    Check if the text has sufficient vocabulary diversity.
    
    Returns True if the ratio of unique words is below the minimum threshold.
    """
    words = text.split()
    if len(words) == 0:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < min_ratio


def check_special_char_ratio(text: str, max_ratio: float = 0.1) -> bool:
    """
    Check if the text contains too many special characters.
    
    Returns True if special character ratio exceeds the threshold.
    """
    if len(text) == 0:
        return False
    special_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
    ratio = special_chars / len(text)
    return ratio > max_ratio


def contains_suspicious_pattern(text: str, patterns: list) -> bool:
    """
    Check if the text contains any suspicious repetitive patterns.
    
    Returns True if any pattern is found.
    """
    for pattern in patterns:
        if pattern in text:
            return True
    return False


def check_conversations_repetition(conversations, repeat_threshold: float = 0.4, ngram: int = 10) -> bool:
    """
    Check if any model answer in a conversation has an n-gram repetition ratio above `repeat_threshold`.
    
    `conversations` should be a list of dicts with keys 'from' and 'value'.
    Returns True if repetition is detected in any GPT answer.
    """
    for conversation in conversations:
        if conversation['from'] == 'gpt':
            model_answer = conversation['value']
            repeat_ratio = calculate_ngram_repetition(model_answer, ngram)
            if repeat_ratio > repeat_threshold:
                return True
    return False


def flag_function_1(answer: str, super_long_words: int = 25, tail_len: int = 15, tail_count: int = 5) -> bool:
    """
    Flag if the answer contains a super long sentence AND tail repetition.
    This often indicates a 'looping' output.
    
    Args:
        answer: The answer text to check
        super_long_words: Number of words to consider a sentence super long (default 25)
        tail_len: Length of tail to check for repetition (default 15)
        tail_count: Number of times tail must repeat (default 5)
    """
    if len(answer) < tail_len:
        return False
    flag = is_super_long_sentence(answer, n=super_long_words)
    flag2 = answer.count(answer[-tail_len:]) >= tail_count
    return flag and flag2


def flag_function_2(answer: str, extreme_long_words: int = 45) -> bool:
    """
    Flag if the answer contains an extremely long sentence.
    
    Args:
        answer: The answer text to check
        extreme_long_words: Number of words to consider a sentence extremely long (default 45)
    """
    return is_super_long_sentence(answer, n=extreme_long_words)


def flag_function_3(answer: str, search_string: str) -> bool:
    """
    Flag if the answer contains a predefined suspicious string pattern.
    Commonly used for detecting fixed repetitive sequences (like 0000... or counting sequences).
    
    Args:
        answer: The answer text to check
        search_string: The suspicious pattern to search for
    """
    return search_string in answer


def flag_function_4(answer: str, tail_len: int = 15, tail_count: int = 5) -> bool:
    """
    Flag if the last N characters of the answer are repeated multiple times.
    This catches tail repetition loops.
    
    Args:
        answer: The answer text to check
        tail_len: Length of tail to check (default 15)
        tail_count: Number of times tail must repeat (default 5)
    """
    if len(answer) < tail_len:
        return False
    return answer.count(answer[-tail_len:]) >= tail_count


def flag_function_5(answer: str, min_unique_ratio: float = 0.3) -> bool:
    """
    Flag if the answer has insufficient vocabulary diversity.
    
    Args:
        answer: The answer text to check
        min_unique_ratio: Minimum ratio of unique words required (default 0.3)
    """
    return check_unique_word_ratio(answer, min_unique_ratio)


def flag_function_6(answer: str, max_special_ratio: float = 0.1) -> bool:
    """
    Flag if the answer contains too many special characters.
    
    Args:
        answer: The answer text to check
        max_special_ratio: Maximum allowed ratio of special characters (default 0.1)
    """
    return check_special_char_ratio(answer, max_special_ratio)


def flag_function_7(answer: str, suspicious_patterns: list) -> bool:
    """
    Flag if the answer contains any suspicious repetitive patterns.
    
    Args:
        answer: The answer text to check
        suspicious_patterns: List of patterns to check for
    """
    return contains_suspicious_pattern(answer, suspicious_patterns)

