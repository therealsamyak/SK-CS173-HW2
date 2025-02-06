import math

from data_tokenization import DataPoint, read_and_process_file
from sentence_likelihood import classify_sentence
from word_likelihood import word_likelihood


class Metrics:
    def __init__(self, emotion, accuracy=0, precision=0, recall=0, f1_score=0):
        self.emotion = emotion
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score

    def __str__(self):
        return (
            f"For emotion {self.emotion}:\n"
            f"Accuracy: {self.accuracy}\n"
            f"Precision: {self.precision}\n"
            f"Recall: {self.recall}\n"
            f"F1 Score: {self.f1_score}"
        )


def sent_likelihood_from_tokens(sentence_tokens: list[str], emotion: str):
    prob = 0

    for word in sentence_tokens:
        word_prob = word_likelihood(word, emotion)
        prob += math.log(word_prob)

    return prob


def confusion_matrix(data: list[DataPoint], emotions: list[str]):
    conf_matrix = [[0 for _ in range(6)] for _ in range(6)]

    for obj in data:
        true_label = emotions.index(obj.emotion)
        predicted_label = emotions.index(
            classify_sentence(" ".join(obj.tokens), emotions)
        )
        conf_matrix[predicted_label][true_label] += 1

    return conf_matrix


def print_matrix(matrix: list[list[int]], emotions: list[str]):
    print("\n                ", end="")
    for emotion in emotions:
        print(f"{emotion:<10}", end=" ")
    print()

    for i, row in enumerate(matrix):
        print(f"{emotions[i]:<10}", end="")
        print(" ".join(f"{x:10}" for x in row))


def calc_metrics(conf_matrix, emotions, emotion):
    total_samples = sum(sum(row) for row in conf_matrix)

    i = emotions.index(emotion)
    true_pos = conf_matrix[i][i]
    false_pos = sum(conf_matrix[row][i] for row in range(len(emotions)) if row != i)
    false_neg = sum(conf_matrix[i][col] for col in range(len(emotions)) if col != i)
    true_neg = total_samples - (true_pos + false_pos + false_neg)

    metrics = Metrics(emotion)

    metrics.accuracy = 1.0 * (true_pos + true_neg) / total_samples
    metrics.precision = 1.0 * true_pos / (true_pos + false_pos)
    metrics.recall = 1.0 * true_pos / (true_pos + false_neg)
    metrics.f1_score = (
        2.0
        * (metrics.precision * metrics.recall)
        / (metrics.precision + metrics.recall)
    )

    return metrics


if __name__ == "__main__":
    test_data = read_and_process_file()[2]
    emotions = ["Joy", "Fear", "Anger", "Surprise", "Disgust", "Sadness"]

    conf_matrix = confusion_matrix(test_data, emotions)
    print_matrix(conf_matrix, emotions)

    metrics = calc_metrics(conf_matrix, emotions, "Joy")
    print()
    print(metrics)
