from data_tokenization import read_and_process_file

train_data = read_and_process_file()[0]

# filter data
joy_data = list(filter(lambda obj: obj.emotion == "Joy", train_data))
fear_data = list(filter(lambda obj: obj.emotion == "Fear", train_data))
anger_data = list(filter(lambda obj: obj.emotion == "Anger", train_data))
surprise_data = list(filter(lambda obj: obj.emotion == "Surprise", train_data))
disgust_data = list(filter(lambda obj: obj.emotion == "Disgust", train_data))
sadness_data = list(filter(lambda obj: obj.emotion == "Sadness", train_data))

# calc priors
data_len = len(train_data)
prior_joy = len(joy_data) / data_len
prior_fear = len(fear_data) / data_len
prior_anger = len(anger_data) / data_len
prior_surprise = len(surprise_data) / data_len
prior_disgust = len(disgust_data) / data_len
prior_sadness = len(sadness_data) / data_len

if __name__ == "__main__":
    print(f"P(Joy): {prior_joy}")
    print(f"P(Fear): {prior_fear}")
    print(f"P(Anger): {prior_anger}")
    print(f"P(Surprise): {prior_surprise}")
    print(f"P(Disgust): {prior_disgust}")
    print(f"P(Sadness): {prior_sadness}")
