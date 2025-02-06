import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from data_tokenization import DataPoint, read_and_process_file
from self_evaluation import calc_metrics, print_matrix


def train_naive_bayes_model(data: list[DataPoint], emotions: list[str]):
    sentences = [" ".join(obj.tokens) for obj in data]
    labels = [obj.emotion for obj in data]
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(sentences, labels)
    return model


def confusion_matrix_with_sklearn(data: list[DataPoint], emotions: list[str], model):
    conf_matrix = np.zeros((len(emotions), len(emotions)), dtype=int)
    sentences = [" ".join(obj.tokens) for obj in data]
    true_labels = [obj.emotion for obj in data]
    predicted_labels = model.predict(sentences)
    for true, predicted in zip(true_labels, predicted_labels):
        true_index = emotions.index(true)
        predicted_index = emotions.index(predicted)
        conf_matrix[predicted_index][true_index] += 1
    return conf_matrix


if __name__ == "__main__":
    test_data = read_and_process_file()[2]
    emotions = ["Joy", "Fear", "Anger", "Surprise", "Disgust", "Sadness"]

    nb_model = train_naive_bayes_model(test_data, emotions)

    conf_matrix = confusion_matrix_with_sklearn(test_data, emotions, nb_model)
    print_matrix(conf_matrix, emotions)

    metrics = calc_metrics(conf_matrix, emotions, "Joy")
    print()
    print(metrics)
