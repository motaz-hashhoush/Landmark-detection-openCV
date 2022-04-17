import os
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from keras.models import model_from_json

def load_data(validation_split):
    # Create path to csv file
    cwd = os.getcwd()
    csv_path = os.path.join(cwd, "data/training.csv")

    # Load data from csv file into data frame, drop all rows that have missing values
    data_frame = read_csv(csv_path)
    print(data_frame["Image"].count())
    data_frame = data_frame.dropna()
    print(data_frame["Image"].count())

    # Convert the rows of the image column from pixel values separated by spaces to numpy arrays
    data_frame["Image"] = data_frame["Image"].apply(lambda img: np.fromstring(img, sep=" "))

    # Create numpy matrix from image column by stacking the rows vertically
    X_data = np.vstack(data_frame["Image"].values)
    # Normalize pixel values to (0, 1) range
    X_data = X_data / 255
    # Convert to float32, which is the default for Keras
    X_data = X_data.astype("float32")
    # Reshape each row from one dimensional arrays to (height, width, num_channels) = (96, 96, 1)
    X_data = X_data.reshape(-1, 96, 96, 1)
    # Extract labels representing the coordinates of facial landmarks
    Y_data = data_frame[data_frame.columns[:-1]].values

    # Normalize coordinates to (0, 1) range
    Y_data = Y_data / 96
    Y_data = Y_data.astype("float32")

    # Shuffle data
    X_data, Y_data = shuffle(X_data, Y_data)

    # Split data into training set and validation set
    split_index = int(X_data.shape[0] * (1 - validation_split))
    X_train = X_data[:split_index]
    Y_train = Y_data[:split_index]
    X_val = X_data[split_index:]
    Y_val = Y_data[split_index:]

    return X_train, Y_train, X_val, Y_val

def save_img_with_landmarks(img, landmarks, plot_name, gray_scale=False):
    if gray_scale:
        plt.imshow(np.squeeze(img), cmap=plt.get_cmap("gray"))
    else:
        plt.imshow(np.squeeze(img))
    for landmark in landmarks:
        plt.plot(landmark[0], landmark[1], "go")
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()


def load_model(model_name):
    with open("saved_models/" + model_name + "_model_architecture.json", "r") as f:
        model = model_from_json(f.read())
    model.load_weights("saved_models/" + model_name + "_model_weights.h5")
    return model

def extract_landmarks(y_pred, img_size_x, img_size_y):
    landmarks = []
    for i in range(0, len(y_pred), 2):
        landmark_x, landmark_y = y_pred[i] * img_size_x, y_pred[i+1] * img_size_y
        landmarks.append((landmark_x, landmark_y))
    return landmarks

