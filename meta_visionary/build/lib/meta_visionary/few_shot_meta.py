import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def conv_bn(x):
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def build_prototypical_net(input_shape, num_classes):
    inputs = layers.Input(input_shape)
    x = conv_bn(inputs)
    x = conv_bn(x)
    x = conv_bn(x)
    x = conv_bn(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)

def meta_learning(train_data_dir, test_data_dir, num_classes=2, learning_rate=0.003, meta_step_size=0.25,
                  inner_batch_size=25, eval_batch_size=25, meta_iters=1, eval_iters=5, inner_iters=4,
                  eval_interval=1, train_shots=20, shots=5, img_height=28, img_width=28):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    model = build_prototypical_net((img_height, img_width, 3), num_classes)

    learning_rate = 0.001
    optimizer = optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    training = []
    testing = []

    curr = time.time()
    for meta_iter in range(meta_iters):
        frac_done = meta_iter / meta_iters
        cur_meta_step_size = (1 - frac_done) * meta_step_size
        old_vars = model.get_weights()
        mini_dataset = iter(train_generator)
        for i in range(inner_iters):
            images, labels = next(mini_dataset)
            with tf.GradientTape() as tape:
                preds = model(images)
                loss = tf.keras.losses.categorical_crossentropy(labels, preds)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        new_vars = model.get_weights()
        for var in range(len(new_vars)):
            new_vars[var] = old_vars[var] + ((new_vars[var] - old_vars[var]) * cur_meta_step_size)
        model.set_weights(new_vars)
        if meta_iter % eval_interval == 0:
            accuracies = []
            for dataset in (train_data_dir, test_data_dir):
                data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
                dataset = data_gen.flow_from_directory(
                    dataset,
                    target_size=(img_height, img_width),
                    batch_size=eval_batch_size,
                    class_mode='categorical',
                    shuffle=False
                )
                images, labels = next(dataset)
                old_vars = model.get_weights()
                for i in range(inner_iters):
                    with tf.GradientTape() as tape:
                        preds = model(images)
                        loss = tf.keras.losses.categorical_crossentropy(labels, preds)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                test_preds = model.predict(images)
                test_preds = tf.argmax(test_preds, axis=1).numpy()
                num_correct = (test_preds == tf.argmax(labels, axis=1).numpy()).sum()
                model.set_weights(old_vars)
                accuracies.append(num_correct / eval_batch_size)
            training.append(accuracies[0])
            testing.append(accuracies[1])
            if meta_iter % 100 == 0:
                print("batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1]))

    stop=time.time()
    test_accuracy = testing[-1]

    test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_data_gen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    true_labels = test_generator.classes
    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)

    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    f1=2 * ((precision * recall) / (precision + recall))

    print(f'Final Test accuracy: {test_accuracy * 100:.2f}%')
    print(f'Final Time : {stop-curr}')
    print(f'Final Precision: {precision}')
    print(f'Final Recall: {recall}')
    print(f'Final F1-score: {f1}')


