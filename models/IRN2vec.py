from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import io

vocab_size = 10000

# TODO(leo bright) how to load dataset
imdb = keras.datasets.imdb
# TODO(leo bright) train_data and test_data are integer index, attention to the type
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# TODO(leo bright) A dictionary mapping words to an integer index
word_index={}
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

embedding_dim=128


class IRN2vec(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, input_length=2):
        super(IRN2vec, self).__init__(name='')
        self.embedding = layers.Embedding(vocab_size, embedding_dim, input_length=input_length),  # input_length is the node number in each samples
        self.pooling1D = layers.GlobalAveragePooling1D(),   # TODO(leo bright) instead of compute layer which compute the mul(x, y) value.
        self.fc1 = layers.Dense(16, activation='relu'),
        self.fc2 = layers.Dense(1, activation='sigmoid')

    def call(self, input_tensor, training=False):
        x = self.embedding(input_tensor)
        x = self.pooling1D(x, training=training)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        return tf.nn.relu(x)


# model = keras.Sequential([
#   layers.Embedding(vocab_size, embedding_dim, input_length=2),  # input_length is the node number in each samples
#   layers.GlobalAveragePooling1D(),   # TODO(leo bright) instead of compute layer which compute the mul(x, y) value.
#   layers.Dense(16, activation='relu'),
#   layers.Dense(1, activation='sigmoid')
# ])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data,
    train_labels,
    epochs=30,
    batch_size=512,
    validation_split=0.2)

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.show()

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word_num in range(vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")

out_v.close()
out_m.close()
