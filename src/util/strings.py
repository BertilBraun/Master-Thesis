from collections import Counter


def text_similarity(a: str, b: str) -> float:
    # count the words in a that are in b including the number of times they appear -> all words of a are in b -> 1
    c = Counter(a.split())
    d = Counter(b.split())
    if len(c) == 0:
        return 0
    return sum((c & d).values()) / sum(c.values())
