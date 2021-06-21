import numpy as np
import tensorflow as tf
import json
import operator
from tensorflow import keras

NUM_MACD_REF_DAYS = 20
LONG_TERM_DAYS = 200
SHORT_TERM_DAYS = 30
VALIDATION_DATASET_RATIO = 0.3
EPOCHS = 500
CATEGORY_SIZE = 5


def load_json(path):
    with open(path) as f:
        data = json.load(f)
        data = sorted(data, key=lambda x: x['date'])
    return np.array(data)

def build_model():
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True), input_shape=(NUM_MACD_REF_DAYS, NUM_MACD_REF_DAYS))
    )
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(32)))
    # model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(CATEGORY_SIZE ,activation="softmax"))

    model.summary()
    model.compile(
        # optimizer=keras.optimizers.Adam(),
        optimizer=keras.optimizers.RMSprop(),
        # Loss function to minimize
        loss=keras.losses.CategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Accuracy()],
    )
    return model

def get_y_category(reward):
    _cate = 0
    if reward > 2:
        _cate = 4
    elif reward > 0.5:
        _cate = 3
    elif reward < -0.5:
        _cate = 2
    elif reward < -2:
        _cate = 1
    # reward = int(reward+4)
    # reward = max(reward, 1)
    # reward = min(reward, 8)
    return tf.one_hot(_cate, depth=CATEGORY_SIZE)


def train_model(model, datarray):
    _train_x = []
    _train_y = []
    _val_x = []
    _val_y = []
    _prices = np.array([_['price'] for _ in datarray])
    _long_avgs = []
    _short_avgs = []
    # for _data in datarray[LONG_TERM_DAYS:]:
    for idx in range(_prices.shape[0]):
        if idx < LONG_TERM_DAYS:
            continue
        _long_avg = _prices[idx-LONG_TERM_DAYS:idx].sum() / LONG_TERM_DAYS
        _short_avg = _prices[idx-SHORT_TERM_DAYS:idx].sum() / SHORT_TERM_DAYS
        _reward_ratio = (_prices[idx] - _prices[idx-1]) / _prices[idx-1] * 100
        _category = get_y_category(_reward_ratio)
        _long_avgs.append(_long_avg)
        _short_avgs.append(_short_avg)
        if len(_long_avgs) > NUM_MACD_REF_DAYS:
            _la = np.array(_long_avgs[-NUM_MACD_REF_DAYS:])
            _sa = np.array(_short_avgs[-NUM_MACD_REF_DAYS:])
            _xx = (_la - _la.mean(), _sa - _sa.mean())
            _train_x.append(_xx)
            _train_y.append(_category)

    _num_val_split = int(len(_train_x) * VALIDATION_DATASET_RATIO)

    _val_x = np.array(_train_x[-_num_val_split:])
    _val_y = np.array(_train_y[-_num_val_split:])
    _train_x =  np.array(_train_x[:-_num_val_split])
    _train_y =  np.array(_train_y[:-_num_val_split])

    print('_train_x: ', _train_x)
    print('_train_y: ', _train_y)
    np.save('_val_x.npy', _val_x)
    np.save('_val_y.npy', _val_y)
    np.save('_train_x.npy', _train_x)
    np.save('_train_y.npy', _train_y)

    history = model.fit(
        _train_x,
        _train_y,
        # batch_size=64,
        epochs=EPOCHS,
        validation_data=(_val_x, _val_y),
    )
    
    # loss, acc = model.evaluate(_val_x, _val_y, verbose=2)
    # print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
    predictions = model.predict(_train_x)

    total = len(predictions)
    right = 0
    for pidx in range(total):
        pp = predictions[pidx]
        # ans = _val_y[pidx]
        ans = _train_y[pidx]
        if np.argmax(pp) == np.argmax(ans):
            right += 1
    
    print("first accuracy: {:5.2f}%".format(100 * right / total))

    return model, history


if __name__ == '__main__':
    ixic_data = load_json('./ixic.json')
    print(ixic_data[:3])
    model = build_model()
    model, history = train_model(model, ixic_data)
    model.save('saved_model.h5')

    _val_x = np.load('_val_x.npy')
    _val_y = np.load('_val_y.npy')
    _train_x = np.load('_train_x.npy')
    _train_y = np.load('_train_y.npy')
    
    # predictions = model.predict(_val_x)
    predictions = model.predict(_train_x)

    total = len(predictions)
    right = 0
    for pidx in range(total):
        pp = predictions[pidx]
        # ans = _val_y[pidx]
        ans = _train_y[pidx]
        if np.argmax(pp) == np.argmax(ans):
            right += 1
    
    print("second accuracy: {:5.2f}%".format(100 * right / total))
    

