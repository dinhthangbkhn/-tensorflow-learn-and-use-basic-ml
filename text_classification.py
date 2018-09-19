import tensorflow as tf
from tensorflow import keras
imdb = keras.datasets.imdb

#PREPROCESS THE DATA
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data.shape)
word_index=imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#This function will convert the integer number vector to sentences
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#each example in train_data(sentence) will be a vector such that max length is 256, if its length is smaller than 256 then 256-length PAD will be inserted
train_data=keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=256)
test_data=keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=256)

#BUILD THE MODEL
vocab_size=10000
model = keras.Sequential()

#takes the integer-encoded vocabulary and looks up the embedding vector for each word-index
model.add(keras.layers.Embedding(vocab_size, 16))

#returns a fixed-length output vector for each example by averaging over the sequence dimension
model.add(keras.layers.GlobalAveragePooling1D())

model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.compile(optimizer=tf.train.AdamOptimizer(), loss="binary_crossentropy", metrics=["accuracy"])
x_val=train_data[:10000]
partial_x_train=train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history=model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val), verbose=2)
results = model.evaluate(test_data, test_labels)
print(results)
model.summary()


