import ast
import json
import os
import string
import time
import random
import traceback
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, LSTM, Activation, TimeDistributed, Dropout
from keras.models import Sequential
from keras.optimizers import adam
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException


def request_(req_url, sleep_time=1):
    succeeded = False
    while not succeeded:
        try:
            print("Requesting: %s" % req_url)
            request = Request(req_url)
            request.add_header('User-Agent',
                               'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36')
            response = urlopen(request)
            out = response.read().decode(
                'utf8')  # cos python3 is kind of stupid http://stackoverflow.com/questions/6862770/python-3-let-json-object-accept-bytes-or-let-urlopen-output-strings
            time.sleep(sleep_time)  # obey api rate limits
            succeeded = True
        except:
            sleep_time += 1
            traceback.print_exc()
            continue
    return out


def get_data():
    driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
    driver.set_page_load_timeout(10)
    try:
        driver.get("http://www.allthelyrics.com/lyrics/dmx")
    except TimeoutException:
        pass
    data_str = ""
    hrefs = []
    passed = False
    while not passed:
        try:
            links = driver.find_elements_by_xpath('//div[@class="artist-lyrics-list artist-lyrics-list-all"]//a')
        except TimeoutException:
            continue
        passed = True
    for link in links:
        print(link.get_attribute("href"))
        href = link.get_attribute("href")
        if "javascript:void(0)" not in href:
            hrefs.append(link.get_attribute("href"))
    for href in hrefs:
        try:
            driver.get(href)
        except TimeoutException:
            pass
        time.sleep(3)
        passed = False
        while not passed:
            try:
                content = driver.find_element_by_xpath('//div[@class="content-text-inner"]')
            except (TimeoutException, NoSuchElementException) as e:
                print(e)
                continue
            passed = True
        data_str += content.text
        data_str += "\n"
    return data_str

if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")) as f:
        data = f.read().lower()
else:
    data = get_data()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt"), "w+") as f:
        f.write(str(data))
        
chars = list(set(data))
VOCAB_SIZE = len(chars)  # number of chars available in captains mode essentially
SEQ_LENGTH = 50
num_sequences = int(len(data) / SEQ_LENGTH)

ix_to_char = {ix: char for ix, char in enumerate(chars)}
char_to_ix = {char: ix for ix, char in enumerate(chars)}


def generate_rap(model):
    ix = [char_to_ix[random.choice(string.ascii_letters).lower()]]
    y_chars = [ix_to_char[ix[-1]]]
    X = np.zeros((1, SEQ_LENGTH, VOCAB_SIZE))
    for i in range(SEQ_LENGTH):
        X[0, i, :][ix[-1]] = 1
        ix = np.argmax(model.predict(X[:, :i + 1, :])[0], 1)
        y_chars.append(ix_to_char[ix[-1]])
    return "".join(y_chars)


def plot_learning_curves(hist, filename):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    ax0.set(xlabel="epochs", ylabel="accuracy")
    ax1.set(xlabel="epochs", ylabel="mse")
    ax0.plot(hist['acc'], label="train")
    ax0.plot(hist['val_acc'], label="val")
    ax1.plot(hist['mean_squared_error'], label="train")
    ax1.plot(hist['val_mean_squared_error'], label="val")
    ax0.legend(loc='upper left')
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs/%s.png" % filename))


def rnn(nodes1, nodes2, nodes3, dropout1, dropout2, dropout3, epochs=200, learning_rate=0.001, batch_size=16):
    # 0.001 is default for adam
    X = np.zeros((num_sequences, SEQ_LENGTH, VOCAB_SIZE))
    y = np.zeros((num_sequences, SEQ_LENGTH, VOCAB_SIZE))
    seq_counter = 0
    tmp = list(range(0, num_sequences))
    random.shuffle(tmp)
    for i in tmp:
        seq_counter += 1
        X_sequence = data[i * SEQ_LENGTH:(i + 1) * SEQ_LENGTH]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
        for j in range(SEQ_LENGTH):
            input_sequence[j][X_sequence_ix[j]] = 1.
        X[i] = input_sequence

        y_sequence = data[i * SEQ_LENGTH + 1:(i + 1) * SEQ_LENGTH + 1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))

        for j in range(SEQ_LENGTH):
            target_sequence[j][y_sequence_ix[j]] = 1.
        y[i] = target_sequence

    validation_SEQ_LENGTH = int(num_sequences * 0.2)
    X, Xval = X[:-validation_SEQ_LENGTH], X[-validation_SEQ_LENGTH:]
    y, yval = y[:-validation_SEQ_LENGTH], y[-validation_SEQ_LENGTH:]

    model = Sequential()
    if dropout1:
        model.add(Dropout(dropout1, input_shape=(None, VOCAB_SIZE)))
    model.add(LSTM(nodes1, input_shape=(None, VOCAB_SIZE), return_sequences=True))
    if dropout2:
        model.add(Dropout(dropout2))
    model.add(LSTM(nodes2, return_sequences=True))
    if dropout3:
        model.add(Dropout(dropout3))
    if nodes3:
        model.add(LSTM(nodes3, return_sequences=True))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    optimizer = adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])

    for i, e in enumerate(range(epochs)):
        print("Epoch number %s" % (i + 1))
        results = model.fit(X, y, validation_data=(Xval, yval), verbose=2, epochs=1, batch_size=batch_size)
        for i in range(5):
            print(generate_rap(model))
        model.save_weights('weights2.hdf5')
        out = {
            'mse': results.history["mean_squared_error"],
            'val_mse': results.history["val_mean_squared_error"],
            'accuracy': results.history["acc"],
            'val_accuracy': results.history["val_acc"],
            'nodes1': nodes1,
            'nodes2': nodes2,
            'nodes3': nodes3,
            'dropout1': dropout1,
            'dropout2': dropout2,
            'dropout3': dropout3,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
        }
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w+") as f:
            json.dump(out, f)

    plot_learning_curves(results.history, "rnn(100, 100, 20, 0, 0, 0, epochs=1000)")

if __name__ == "__main__":
    rnn(200, 200, 100, 0.4, 0.3, 0.3, epochs=1000, learning_rate=0.01)
