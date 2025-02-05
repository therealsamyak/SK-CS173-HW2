from datatokenization import DataPoint, read_and_process_file

train_data = read_and_process_file()[0]


def get_vocab_size(data: list[DataPoint]):
    vocab = set()
    for obj in data:
        vocab.update(obj.tokens)

    return len(vocab)


def count_tokens(data: list[DataPoint]):
    return sum(len(obj.tokens) for obj in data)


def count_word(word: str, data: list[DataPoint]):
    return sum(obj.tokens.count(word) for obj in data)


def word_likelihood(word, emotion):

    filtered_data = list(filter(lambda obj: obj.emotion == emotion, train_data))

    # laplace smoothening
    numer = 1 + count_word(word, filtered_data)
    denom = count_tokens(filtered_data) + get_vocab_size(train_data)

    return 1.0 * numer / denom


if __name__ == "__main__":

    word = input("Enter Word: ")
    emotion = input("Enter Emotion (ex. Joy): ").capitalize()

    print(
        f"Probability of '{word}' in '{emotion}' is: {word_likelihood(word, emotion)}"
    )
