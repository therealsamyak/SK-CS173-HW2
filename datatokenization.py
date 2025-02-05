import csv

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


class DataPoint:
    def __init__(self, tokens: list[str], emotion: str):
        self.tokens = tokens
        self.emotion = emotion

    def __str__(self):
        return f"Emotion: {self.emotion}\nTokens: {self.tokens}"


def split_data(data):
    train_end = 30
    val_end = train_end + 9

    train_data = [DataPoint(row[2], row[1]) for row in data if int(row[0]) <= train_end]
    val_data = [
        DataPoint(row[2], row[1]) for row in data if train_end < int(row[0]) <= val_end
    ]
    test_data = [DataPoint(row[2], row[1]) for row in data if int(row[0]) > val_end]

    return train_data, val_data, test_data


def read_and_process_file():
    data = []

    with open("cs173-hw2-processed.csv", "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # skip header

        for row in reader:
            row_num, emotion, text = row
            tokens = nltk.word_tokenize(text)
            data.append([row_num, emotion, tokens])

    return split_data(data)


train_data, val_data, test_data = read_and_process_file()

if __name__ == "__main__":
    for i in range(10):
        print(train_data[i])
