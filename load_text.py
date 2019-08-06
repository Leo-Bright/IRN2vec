from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import os

DIRECTORY_URL = 'datasets'
FILE_NAME = 'sanfrancisco_random_1w.walks'


def labelor(example, sep, index):
    new_example = []
    items = example.split(sep)
    for i in range(len(items)):
        if i % index == 0:
            new_example.append(items[i])

    return format(' ').join(new_example), tf.cast(label, tf.int64)


labeled_data_sets = []
lines_dataset = tf.data.TextLineDataset(os.path.join(DIRECTORY_URL, FILE_NAME))
all_split_dataset = lines_dataset.map(lambda ex: labelor(ex, ' ', 2))

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_split_shuffled_data = all_split_dataset.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)


tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_split_shuffled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


all_encoded_data = all_split_shuffled_data.map(encode_map_fn)

train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

# sample_text, sample_labels = next(iter(test_data))
# #
# # sample_text[0], sample_labels[0]

vocab_size += 1

model = tf.keras.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, 64))

model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

# One or more dense layers.
# Edit the list in the `for` line to experiment with layer sizes.
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))