import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

check_path = "checkpoint/cp.ckpt"
check_dir = os.path.dirname(check_path)

boston_housing = keras.datasets.boston_housing

#PREPROCESSING DATA
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

def preprocess_dataset(dataset):
    means = dataset.mean(axis=0)
    std = dataset.std(axis=0)
    return (dataset-means)*1.0/std
train_data = preprocess_dataset(train_data)
test_data = preprocess_dataset(test_data)

fix=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])
train_data=train_data*fix
test_data=test_data*fix

#BUILD MODEL
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=64,activation=tf.nn.relu, input_shape=(train_data.shape[1],), kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    model.add(keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    model.add(keras.layers.Dense(1))
    optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
    return model

model = build_model()
print("Untrained model:")
loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)

save_model = keras.callbacks.ModelCheckpoint(filepath=check_path, save_weights_only=True, verbose=1)
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
model.fit(train_data, train_labels, epochs=500, batch_size=128, validation_split=0.2, callbacks=[ save_model])
result = model.evaluate(test_data, test_labels)
print(result)
model.load_weights(check_path)
print("Load model from checkpoint")
print(model.evaluate(test_data, test_labels))

