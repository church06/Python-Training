import tensorflow as tf
import tf_Model

print('GPUs: ', len(tf.config.list_physical_devices('GPU')))
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10)
# ])
#
# predictions = model(x_train[:1]).numpy()
# print(tf.nn.softmax(predictions).numpy())
#
# probability_model = tf.keras.Sequential([
#     model,
#     tf.keras.layers.Softmax()
# ])
#
# print(probability_model(x_test[:5]))

x_train = x_train[..., tf.newaxis].astype('float32')
x_test = x_test[..., tf.newaxis].astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

model = tf_Model.MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


def train_step(image, label):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).

        predictions = model(image, training=True)
        loss = loss_object(label, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(label, predictions)


def test_step(image, label):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(image, training=False)
    t_loss = loss_object(label, predictions)

    test_loss(t_loss)
    test_accuracy(label, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        'Epoch {0}, '.format(epoch + 1),
        'Loss: {0}, '.format(train_loss.result()),
        'Accuracy: {0}, '.format(train_accuracy.result() * 100),
        'Test Loss: {0}, '.format(test_loss.result()),
        'Test Accuracy: {0}'.format(test_accuracy.result() * 100),
    )
