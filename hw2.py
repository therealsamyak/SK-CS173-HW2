import random
from collections import Counter

import nltk
from nltk.util import ngrams

nltk.download("punkt")
nltk.download("punkt_tab")

random.seed(0)


def split_data(data):

    random.shuffle(data)

    # "midpoint 1"
    train_end = 30

    # "midpoint" 2
    val_end = train_end + 9

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def read_and_process_file():
    with open("cs173_nlp_movie.txt", "r", encoding="utf-8") as file:
        file_content = file.read()
    file_content = file_content.lower()
    tokens = nltk.word_tokenize(file_content)
    train_data, val_data, test_data = split_data(tokens)
    return train_data, val_data, test_data


def generate_n_grams(n: int, text: str) -> str:
    return list(
        ngrams(
            text,
            n,
            pad_left=True,
            left_pad_symbol="<s>",
            pad_right=True,
            right_pad_symbol="</s>",
        )
    )


def count_ngram(n: int, text: list, smoothing=1):
    n_grams = generate_n_grams(n, text)
    ngram_counts = Counter(n_grams)

    vocab_size = len(set(text))

    smoothed_ngram_counts = Counter(
        {ngram: count + smoothing for ngram, count in ngram_counts.items()}
    )

    return smoothed_ngram_counts, vocab_size


def calculate_ngram_probabilities(
    n: int, text: list, ngram_counts, vocab_size, smoothing=1
):
    ngram_probs = {}

    for ngram, count in ngram_counts.items():
        prefix = ngram[:-1]
        prefix_count = Counter(generate_n_grams(n - 1, text))[prefix]

        smoothed_count = count
        smoothed_prefix_count = prefix_count + (vocab_size * smoothing)

        probability = smoothed_count + 1 / smoothed_prefix_count
        ngram_probs[ngram] = probability

    return ngram_probs


n = 3
train_data, val_data, test_data = read_and_process_file()

ngram_counts, vocab_size = count_ngram(n, train_data)
ngram_probs = calculate_ngram_probabilities(n, train_data, ngram_counts, vocab_size)
