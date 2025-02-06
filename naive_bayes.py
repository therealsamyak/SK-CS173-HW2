import math

import nltk

from data_tokenization import read_and_process_file
from priors import calc_priors
from word_likelihood import word_likelihood


def sentence_likelihood(sentence: str, emotion: str):
    sentence_tokens = nltk.word_tokenize(sentence)

    # start with prior prob
    prob = math.log(calc_priors()[emotion])

    for word in sentence_tokens:
        word_prob = (
            word_likelihood(word, emotion) if word_likelihood(word, emotion) > 0 else 1
        )
        prob += math.log(word_prob)

    return prob


def classify_sentence(sentence: str, emotions: list[str]):
    probs = {}

    for emotion in emotions:
        probs[emotion] = sentence_likelihood(sentence, emotion)

    return max(probs, key=probs.get)


if __name__ == "__main__":

    # Replace with other hardcoded sentence
    sentence = "As she hugged her daughter goodbye on the first day of college, she felt both sad to see her go and joyful knowing that she was embarking on a new and exciting chapter in her life."

    emotions = ["Joy", "Fear", "Anger", "Surprise", "Disgust", "Sadness"]
    for emotion in emotions:
        print(
            f"Probability in '{emotion}' is: {sentence_likelihood(sentence, emotion)}"
        )

    print()
    print(f"Sentence is likely of emotion: '{classify_sentence(sentence, emotions)}'")
