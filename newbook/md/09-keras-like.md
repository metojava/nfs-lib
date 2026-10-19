# FashionMNISTRunner: train in five lines, explain in fifty

*The goal model is the classic Keras starter. This chapter shows the direct Java translation, then walks through the whole runner with every constant, parameter and log line explained.*

## 9.1 The Keras original

```
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```

## 9.2 The Java translation

```
// Flatten(input_shape=(28,28))  -> an input layer of 784 value-only
//                                 neurons (784 = 28 * 28)
// Dense(128, activation='relu') -> 128 hidden neurons, ReLU
// Dense(10)                     -> 10 output neurons, linear scores

int[] topology = {INPUTS, 128, CLASSES};       // {784, 128, 10}

ActivationFunction[] activations = {
    ActivationFunction.linear(),   // input (identity, no weights)
    ActivationFunction.relu(),     // Dense(128, relu)
    ActivationFunction.linear()    // Dense(10) — plain scores
};

NeuralNetwork network = new NeuralNetwork(
    topology,
    activations,
    LossFunction.meanSquaredError(),
    new Adam(0.001),
    SEED
);
```

> **Mapping table.** Keras `Flatten` = fixed 784-value input layer here (the loader already flattens each image to a 784-vector). Keras `Dense(10)` with no activation = `linear()`. The loss differs: Keras examples often use `from_logits` softmax cross-entropy, while this library's starter uses MSE over linear outputs — that's why argmax, not softmax, picks the class.

### Architecture parameters

| Constant | Value | Meaning |
| --- | --- | --- |
| `INPUTS` | 784 | 28×28 pixels, already normalized to [0,1] by the loader. |
| `CLASSES` | 10 | Fashion categories; also the one-hot width. |
| hidden | 128 | ReLU units — same as Keras' Dense(128). |
| `EPOCHS` | 10 | Full passes over the training set. |
| `BATCH_SIZE` | 128 | Samples per gradient update. |
| `SEED` | 123 | Fixed RNG seed: same weights and same shuffles every run. |
| learning rate | 0.001 | Adam's standard default. |

## 9.3 The main() flow

```
/**
 * Command-line driver for the Fashion-MNIST classifier.
 *
 * @param args optional "--quick" to train on only 8,000 examples
 * @throws Exception on download, file or training errors
 */
public static void main(String[] args) throws Exception {
    boolean quickMode = Arrays.asList(args).contains("--quick");

    Path dataDirectory = Path.of("data", "fashion-mnist");

    // 1. Dataset on disk? download if missing
    FashionMNISTDownloader.download(dataDirectory);

    // 2. Parse IDX into arrays
    FashionMNISTLoader.Dataset training =
        FashionMNISTLoader.loadTraining(dataDirectory);
    FashionMNISTLoader.Dataset testing =
        FashionMNISTLoader.loadTest(dataDirectory);

    // 3. Optional subset for a fast smoke test
    int trainingCount = quickMode
        ? Math.min(8000, training.size())
        : training.size();

    double[][] trainImages = Arrays.copyOf(
        training.getImages(), trainingCount);
    int[] trainLabels = Arrays.copyOf(
        training.getLabels(), trainingCount);

    double[][] trainTargets = oneHotEncode(trainLabels);
    double[][] testImages  = testing.getImages();
    int[] testLabels = testing.getLabels();

    // 4. Build the Keras-like model (see above)

    // 5. Train epoch by epoch, logging loss / accuracy / time
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        Trainer trainer = new Trainer(
            1,               // one epoch per iteration
            BATCH_SIZE,
            true,            // shuffle
            SEED + epoch     // new, reproducible shuffle each epoch
        );

        long epochStart = System.currentTimeMillis();

        network.fit(trainer, trainImages, trainTargets);

        double loss     = network.calculateLoss(trainImages, trainTargets);
        double accuracy = evaluateAccuracy(network, testImages, testLabels);
        long elapsed    = System.currentTimeMillis() - epochStart;

        System.out.printf("%-7d %13.6f %10.2f%% %7.1f s%n",
            epoch, loss, accuracy * 100.0, elapsed / 1000.0);
    }

    // 6. Final evaluation + sample predictions
    evaluate(network, testImages, testLabels);
}
```

### The console output looks like this

```
Training (Keras-like simple model)...
Epoch      Train loss    Accuracy      Time
-------------------------------------------
1            0.048426      73.53%     3.9 s
2            0.039638      77.75%     3.6 s
3            0.036067      79.00%     3.7 s
4            0.032357      81.14%     3.7 s
5            0.030762      81.61%     3.8 s
6            0.028398      82.43%     3.7 s
7            0.027136      83.03%     3.7 s
8            0.025725      83.44%     3.7 s
9            0.024997      83.28%     3.7 s
10           0.024986      83.46%     3.7 s

Training completed in 37.3 s
Final training loss: 0.024986
Test accuracy: 83.46%
```

(Quick mode with 8,000 examples; a full run uses all 60,000 and reaches roughly 87–89%.)

## 9.4 Why train one epoch at a time?

Calling `fit` once with `epochs = 10` would be shorter code, but nothing could be printed in between. Wrapping the call in a loop lets each epoch be timed and evaluated separately — loss, accuracy and wall-clock seconds per epoch, exactly what you want when tuning hyper-parameters. Each epoch gets a fresh `Trainer` with seed `SEED + epoch`, so shuffling stays reproducible but differs between epochs.

## 9.5 Choosing quick vs full mode

```
# quick smoke test (~40 s): 8,000 examples, 10 epochs
java -cp target/classes FashionMNISTRunner --quick

# full training: 60,000 examples, 10 epochs (takes minutes)
java -cp target/classes FashionMNISTRunner
```

---

[← 08 · Downloader & loader](08-downloader-loader.md) · [10 · Evaluation & confusion matrix →](10-evaluation.md)
