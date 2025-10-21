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


def flag_function_1(answer: str) -> bool:
    """
    Flag if the answer contains a super long sentence (>=20 words)
    AND its last 20 characters repeat at least 8 times.
    This often indicates a 'looping' output.
    """
    if len(answer) < 20:
        return False
    flag = is_super_long_sentence(answer, n=20)
    flag2 = answer.count(answer[-20:]) >= 8
    return flag and flag2


def flag_function_2(answer: str) -> bool:
    """
    Flag if the answer contains an extremely long sentence (>=50 words).
    """
    return is_super_long_sentence(answer, n=50)


def flag_function_3(answer: str, search_string: str) -> bool:
    """
    Flag if the answer contains a predefined suspicious string pattern.
    Commonly used for detecting fixed repetitive sequences (like 0000... or counting sequences).
    """
    return search_string in answer


def flag_function_4(answer: str) -> bool:
    """
    Flag if the last 20 characters of the answer are repeated at least 8 times.
    This catches tail repetition loops.
    """
    if len(answer) < 20:
        return False
    return answer.count(answer[-20:]) >= 8

