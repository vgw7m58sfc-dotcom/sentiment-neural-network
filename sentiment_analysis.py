import numpy as np
import tensorflow as tf
import random
from keras import models, layers
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Set seeds for fair comparison
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Load and vectorize the data
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=10000)

def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

train_x = vectorize(train_x)
test_x = vectorize(test_x)
train_y = np.array(train_y).astype("float32")
test_y = np.array(test_y).astype("float32")

# Custom activation functions
def adapted_relu(x):
    return tf.where(x < 0, x / (1 - x), x + (x**2) / (2 + 2*x**2))

def adapted_sigmoid(x):
    return 0.5 * (1 + x / tf.sqrt(1 + x**2))

# Custom callback to record precision, recall, and F1 each epoch
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.val_x, self.val_y = val_data
        self.precisions = []
        self.recalls = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs=None):
        val_pred = (self.model.predict(self.val_x, verbose=0) >= 0.5).astype("float32")
        self.precisions.append(precision_score(self.val_y, val_pred, zero_division=0))
        self.recalls.append(recall_score(self.val_y, val_pred, zero_division=0))
        self.f1s.append(f1_score(self.val_y, val_pred, zero_division=0))

# Build and train the neural network
def build_and_train(activation, activation_name):
    tf.random.set_seed(42)
    np.random.seed(42)

    model = models.Sequential([
        layers.Dense(16, activation=activation, input_shape=(10000,)),
        layers.Dropout(0.5),
        layers.Dense(16, activation=activation),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    metrics_callback = MetricsCallback(val_data=(test_x, test_y))

    print(f"Training with {activation_name}")
    history = model.fit(
        train_x,
        train_y,
        epochs=20,
        batch_size=512,
        validation_data=(test_x, test_y),
        callbacks=[early_stop, metrics_callback],
        verbose=1
    )

    scores = model.evaluate(test_x, test_y, verbose=0)
    print(f"{activation_name} Final Accuracy: {scores[1] * 100:.2f}%")
    print(f"{activation_name} Stopped at epoch: {len(history.history['loss'])}")

    val_pred = (model.predict(test_x, verbose=0) >= 0.5).astype("float32")
    final_precision = precision_score(test_y, val_pred, zero_division=0)
    final_recall = recall_score(test_y, val_pred, zero_division=0)
    final_f1 = f1_score(test_y, val_pred, zero_division=0)

    return history, scores, metrics_callback, final_precision, final_recall, final_f1

# Train all four neural networks
history_relu, scores_relu, cb_relu, p_relu, r_relu, f_relu = build_and_train("relu", "ReLU")
history_sigmoid, scores_sigmoid, cb_sigmoid, p_sigmoid, r_sigmoid, f_sigmoid = build_and_train("sigmoid", "Sigmoid")
history_arelu, scores_arelu, cb_arelu, p_arelu, r_arelu, f_arelu = build_and_train(adapted_relu, "Adapted ReLU")
history_asigmoid, scores_asigmoid, cb_asigmoid, p_asigmoid, r_asigmoid, f_asigmoid = build_and_train(adapted_sigmoid, "Adapted Sigmoid")

# Plotting helper
def plot_metric(histories, callbacks, metric, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    labels = ["ReLU", "Sigmoid", "Adapted ReLU", "Adapted Sigmoid"]

    for hist, cb, label in zip(histories, callbacks, labels):
        if metric == "precision":
            data = cb.precisions
        elif metric == "recall":
            data = cb.recalls
        elif metric == "f1":
            data = cb.f1s
        else:
            data = hist.history[metric]

        plt.plot(range(1, len(data) + 1), data, label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(range(1, max(len(h.history["loss"]) for h in histories) + 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

histories = [history_relu, history_sigmoid, history_arelu, history_asigmoid]
callbacks = [cb_relu, cb_sigmoid, cb_arelu, cb_asigmoid]

# Generate and save all plots
plot_metric(histories, callbacks, "loss", "Training Loss by Activation Function", "Training Loss", "training_loss.png")
plot_metric(histories, callbacks, "val_loss", "Validation Loss by Activation Function", "Validation Loss", "validation_loss.png")
plot_metric(histories, callbacks, "val_accuracy", "Accuracy by Activation Function", "Accuracy", "accuracy.png")
plot_metric(histories, callbacks, "precision", "Precision by Activation Function", "Precision", "precision.png")
plot_metric(histories, callbacks, "recall", "Recall by Activation Function", "Recall", "recall.png")
plot_metric(histories, callbacks, "f1", "F1 Score by Activation Function", "F1 Score", "f1_score.png")

plt.show()

# Summary table
print("\n=== Final Results Summary ===")
print(f"{'Function':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Epochs':>8}")
print("-" * 70)

results = [
    ("ReLU", scores_relu[1], p_relu, r_relu, f_relu, history_relu),
    ("Sigmoid", scores_sigmoid[1], p_sigmoid, r_sigmoid, f_sigmoid, history_sigmoid),
    ("Adapted ReLU", scores_arelu[1], p_arelu, r_arelu, f_arelu, history_arelu),
    ("Adapted Sigmoid", scores_asigmoid[1], p_asigmoid, r_asigmoid, f_asigmoid, history_asigmoid),
]

for name, acc, p, r, f, hist in results:
    print(f"{name:<20} {acc * 100:>9.2f}% {p:>10.4f} {r:>10.4f} {f:>10.4f} {len(hist.history['loss']):>8}")