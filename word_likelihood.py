from data_tokenization import DataPoint, read_and_process_file

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


def word_likelihood(word: str, emotion: str):

    filtered_data = list(filter(lambda obj: obj.emotion == emotion, train_data))

    # laplace smoothening
    numer = 1 + count_word(word, filtered_data)
    denom = count_tokens(filtered_data) + get_vocab_size(train_data)

    return 1.0 * numer / denom


def classify_word(word: str, emotions: list[str]):
    probs = {}

    for emotion in emotions:
        probs[emotion] = word_likelihood(word, emotion)

    return max(probs, key=probs.get)


if __name__ == "__main__":

    emotions = ["Joy", "Fear", "Anger", "Surprise", "Disgust", "Sadness"]

    word = input("Enter Word: ")
    for emotion in emotions:
        print(
            f"Probability of '{word}' in '{emotion}' is: {word_likelihood(word, emotion)}"
        )

    print(
        f"'{word}' is likely of emotion: '{classify_word(word, emotions)}'"
    )
