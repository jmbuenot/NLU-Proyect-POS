import numpy as np
import tensorflow as tf
import keras
from keras import layers

class MyTagger(object):
  def __init__(self):
    self.model = None

  def build_model(self, vocabulary):
    text_vectorizer = layers.TextVectorization(output_mode='int', output_sequence_length=128, standardize=None)
    text_vectorizer.adapt(vocabulary)
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = text_vectorizer(inputs)
    x = layers.Embedding((text_vectorizer.vocabulary_size()+1), 32, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    outputs = layers.TimeDistributed(layers.Dense(19, activation="softmax"))(x)

    self.model = keras.Model(inputs= inputs, outputs= outputs)
    self.model.summary()

tf.config.run_functions_eagerly(True)
  @tf.function   
  def train(self, train_inputs, train_targets, num_epochs):
    train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
    train_ds = train_ds.batch(64)
    self.model.compile(optimizer="Adam",
                        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("\nTraining:\n")
    self.model.fit(train_ds, epochs=num_epochs)

  def evaluate(self, eval_inputs, eval_targets):
    eval_ds = tf.data.Dataset.from_tensor_slices((eval_inputs, eval_targets))
    eval_ds = eval_ds.batch(32)
    print("\nEvaluation:\n")
    self.model.evaluate(eval_ds)

  def predict_conllu(self, test_inputs, corpus):
    results = []
    test_ds = tf.data.Dataset.from_tensor_slices(test_inputs)
    test_ds = test_ds.batch(32)
    print("\nPrediction:\n")
    predictions = self.model.predict(test_ds)
    seq_length = [seq.__len__() for seq in corpus]
    pred_no_padding = [pred[:seq_len] for pred, seq_len in zip(predictions, seq_length)]
    for prediction in pred_no_padding:
        result = []
        for sentence in prediction:
            index = np.argmax(sentence)
            result.append(index)
        results.append(result)
    return results
  
  def predict(self, test_inputs):
    results = []
    test_ds = tf.data.Dataset.from_tensor_slices(test_inputs)
    test_ds = test_ds.batch(32)
    print("\nPrediction:\n")
    predictions = self.model.predict(test_ds)
    seq_length = [len(seq.split()) for seq in test_inputs]
    pred_no_padding = [pred[:seq_len] for pred, seq_len in zip(predictions, seq_length)]
    for prediction in pred_no_padding:
        result = []
        for sentence in prediction:
            index = np.argmax(sentence)
            result.append(index)
        results.append(result)
    return results

  def padding(self, targets):
      padded_targets = tf.keras.utils.pad_sequences(targets, maxlen= 128, padding="post")
      return padded_targets