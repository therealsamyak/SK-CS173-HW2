from data_tokenization import read_and_process_file


def calc_priors(
    emotions=["Joy", "Fear", "Anger", "Surprise", "Disgust", "Sadness"],
    train_data=read_and_process_file()[0],
):
    data_len = len(train_data)
    priors = {}

    for emotion in emotions:
        # filter data
        emotion_data = list(filter(lambda obj: obj.emotion == emotion, train_data))

        priors[emotion] = 1.0 * len(emotion_data) / data_len

    return priors


if __name__ == "__main__":
    train_data = read_and_process_file()[0]
    emotions = ["Joy", "Fear", "Anger", "Surprise", "Disgust", "Sadness"]

    priors = calc_priors(train_data, emotions)
    for emotion in emotions:
        print(f"P({emotion}): {priors[emotion]}")
