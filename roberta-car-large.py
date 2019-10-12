# -*- coding: utf-8 -*-

import codecs
import os

import keras_radam as Radam
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import Model, load_model
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_custom_objects

CONFIG_PATH = 'roeberta_zh_L-24_H-1024_A-16/bert_config_large.json'
CHECKPOINT_PATH = 'roeberta_zh_L-24_H-1024_A-16/roberta_zh_large_model.ckpt'
DICT_PATH = 'roeberta_zh_L-24_H-1024_A-16/vocab.txt'

CONFIG = {
    'max_len': 16,
    'batch_size': 8,
    'epochs': 32,
    'use_multiprocessing': True,
    'model_dir': os.path.join('model_files/bert'),
}


def split_train_test(data, X_name, y_name, train_size=0.85, test_size=None):
    """
    对数据切分成训练集和测试集
    :param data:
    :param X_name:特征列
    :param y_name:标签列
    :param train_size:训练比例
    :return:
    """
    train_data = []
    test_data = []
    if (not train_size) and test_size:
        train_size = 1 - test_size
    for i in range(data.shape[0]):
        if i % 100 < train_size * 100:
            train_data.append([str(data.loc[i][X_name]), data.loc[i][y_name]])
        else:
            test_data.append([str(data.loc[i][X_name]), data.loc[i][y_name]])
    return np.array(train_data), np.array(test_data)


class DataGenerator:

    def __init__(self, data, tokenizer, batch_size=CONFIG['batch_size']):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:CONFIG['max_len']]
                x1, x2 = self.tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class OurTokenizer(Tokenizer):

    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


class BertClassify:
    def __init__(self, train=True):
        if train:
            self.bert_model = load_trained_model_from_checkpoint(config_file=CONFIG_PATH,
                                                                 checkpoint_file=CHECKPOINT_PATH)
            # for layer in self.bert_model.layers[: -CONFIG['trainable_layers']]:
            for layer in self.bert_model.layers:
                layer.trainable = True
        self.model = None
        self.__initial_token_dict()
        self.tokenizer = OurTokenizer(self.token_dict)

    def __initial_token_dict(self):
        self.token_dict = {}
        with codecs.open(DICT_PATH, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

    def train(self, train_data, valid_data):
        """
        训练
        :param train_data:
        :param valid_data:
        :return:
        """
        train_D = DataGenerator(train_data, self.tokenizer)
        valid_D = DataGenerator(valid_data, self.tokenizer)

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x_in = self.bert_model([x1_in, x2_in])
        x_in = Lambda(lambda x: x[:, 0])(x_in)
        d = Dropout(0.5)(x_in)
        p = Dense(1, activation='sigmoid')(d)

        # x = Dense(1, activation='softmax')(d)

        # print('Build model...')
        # model = Sequential()
        # model.add(Embedding(768, 128))
        # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        # model.add(Dense(1, activation='sigmoid'))
        # model.summary()

        save = ModelCheckpoint(
            os.path.join(CONFIG['model_dir'], 'bert.h5'),
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )
        early_stopping = EarlyStopping(
            monitor='val_acc',
            min_delta=0,
            patience=8,
            verbose=1,
            mode='auto'
        )
        callbacks = [save, early_stopping]

        self.model = Model([x1_in, x2_in], p)
        # self.model = multi_gpu_model(self.model, gpus=2)  # 设置使用2个gpu，该句放在模型compile之前
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Radam.RAdam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
        self.model.summary()
        self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=CONFIG['epochs'],
            callbacks=callbacks,
            validation_data=valid_D.__iter__(),
            use_multiprocessing=CONFIG['use_multiprocessing'],
            validation_steps=len(valid_D)
        )

    def predict(self, test_data):
        """
        预测
        :param test_data:
        :return:
        """
        X1 = []
        X2 = []
        for s in test_data:
            x1, x2 = self.tokenizer.encode(first=s[:CONFIG['max_len']])
            X1.append(x1)
            X2.append(x2)
        X1 = seq_padding(X1)
        X2 = seq_padding(X2)
        predict_results = self.model.predict([X1, X2])
        return predict_results

    def load(self, model_dir):
        """
        load the pre-trained model
        """
        model_path = os.path.join(model_dir, 'bert.h5')
        try:
            graph = tf.Graph()
            with graph.as_default():
                session = tf.Session()
                with session.as_default():
                    self.reply = load_model(
                        str(model_path),
                        custom_objects=get_custom_objects(),
                        compile=False
                    )
                    with open(os.path.join(model_dir, 'label_map_bert.txt'), 'r') as f:
                        self.label_map = eval(f.read())
                    self.graph = graph
                    self.session = session
        except Exception as ex:
            print('load error')
        return self


if __name__ == "__main__":
    data = pd.read_csv(os.path.join('data/merged.csv'), encoding='utf-8')
    test_data = pd.read_csv(os.path.join('data/Test_DataSet.csv'), encoding='utf-8')
    train_data, valid_data = split_train_test(data, 'text_a', 'label', train_size=0.8)
    predict_test = []
    for i, j in zip(test_data['title'], test_data['content']):
        if i is not None and j is not None:
            predict_test.append(str(i) + str(j))
    # # # bert
    model = BertClassify(train=True)
    model.train(train_data, valid_data)

    predict_results = model.predict(predict_test)
    with open(os.path.join('data/bert/food-predict.txt'), 'w') as f:
        f.write("id,flag\n")
        for i in range(test_data.shape[0]):
            label = 1 if predict_results[i][0] > 0.5 else 0
            f.write(test_data.id[i] + ',' + str(label) + '\n')
