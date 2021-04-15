import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from tensorflow import keras

training_portion = .99  # Use 80% of data for training, 20% for testing
max_words = 1000        #Max words in text input


data = pd.read_csv('testdata1.csv')
print(data.head())

train_size = int(len(data) * training_portion)

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

train_cat, test_cat = train_test_split(data.iloc[:,1], train_size)  # category is second column
train_text, test_text = train_test_split(data.iloc[:,0], train_size)  # text is first column

tokenize = Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_text) # fit tokenizer to our training text data
x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_cat)
y_train = encoder.transform(train_cat)
y_test = encoder.transform(test_cat)

# Converts the labels to a one-hot representation

num_classes = len(set(y_train))  # set() creates a unique set of objects
#num_classes = np.max(y_train) + 1
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
# Inspect the dimenstions of our training and test data
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# Build the model
layers = keras.layers
models = keras.models
model = models.Sequential()
model.add(layers.Dense(512, input_shape=(max_words,), activation='relu'))  # Hidden layer with 512 nodes
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 32
epochs = 2
history = model.fit(x_train, y_train,       # The variable, history, is normally used to plot learning curves
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Here's how to generate a prediction on individual examples
text_labels = encoder.classes_   #ndarray of output values (labels or classes)
# Examine first 10 test samples of 445
for i in range(len(test_cat)):
    temp = x_test[i]
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]  #predicted class
    print(test_text.iloc[i][:50], "...")                # 50 char sample of text
    print('Actual label:' + test_cat.iloc[i])
    print("Predicted label: " + predicted_label + "\n")

single_test_text = ['I have trouble stopping the car.']


#text_as_nparray = np.array([single_test_text])
text_as_series = pd.Series(single_test_text)

single_x_test = tokenize.texts_to_matrix(text_as_series)
single_prediction = model.predict(np.array([single_x_test]))
single_predicted_label = text_labels[np.argmax(single_prediction)]
print("Predicted label: " + single_predicted_label + "\n")