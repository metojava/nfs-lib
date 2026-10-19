# Understand every line of a Java neural network

*This book explains the actual code in this folder — `Neuron`, `Layer`, `NeuralNetwork`, `Trainer`, `ActivationFunction`, `LossFunction`, `SGD`, `Adam`, and the Fashion-MNIST downloader/loader/runner — with method-by-method documentation, parameter tables, the math behind each step, and complete code examples you can run.*

## What you will learn

- **[01 · Foundations](01-foundations.md)** — Neuron, Layer, the neural-network constructor.
- **[02 · Forward pass](02-forward-pass.md)** — How an input becomes a prediction.
- **[03 · Backpropagation](03-backprop.md)** — Gradients, chain rule, parameter updates.
- **[04 · Activations](04-activations.md)** — Sigmoid, tanh, ReLU, Leaky ReLU, linear.
- **[05 · Losses & optimizers](05-losses-optimizers.md)** — MSE, BCE, SGD, Adam.
- **[06 · Training loop](06-training.md)** — Epochs, mini-batches, shuffling.
- **[07 · Fashion MNIST data](07-data.md)** — IDX format, normalization, one-hot.
- **[08 · Downloader & loader](08-downloader-loader.md)** — Download from GitHub, parse IDX.
- **[09 · Keras-like classifier](09-keras-like.md)** — Flatten → Dense(128, ReLU) → Dense(10).
- **[10 · Evaluation](10-evaluation.md)** — Accuracy, confusion matrix, per-class.
- **[11 · HTML report](11-html-report.md)** — Save results as a web page.
- **[12 · Swing/AWT window](12-swing-ui.md)** — Show images and predictions live.
- **[13 · Run it](13-run.md)** — Compile and run with Maven/CLI.
- **[14 · Full code](14-full-walkthrough.md)** — Complete listing of all classes.
- **[15 · Practice](15-practice.md)** — Exercises to extend the project.
- **[Performance](performance.md)** — float conversion, Vector API, SIMD speedups.
- **[Projects](projects/index.md)** — 12 standalone tutorials: XOR to GAN.

## The plan

**Read** (the code) → **Understand** (the math) → **Download** (Fashion MNIST) → **Train** (the classifier) → **Evaluate** (& report)

> **Prerequisites.** JDK 17, a text editor, and basic Java (arrays, loops, classes). You do not need TensorFlow or any ML library — everything is plain Java. Maven is optional but convenient for running the examples.

## Project layout

```
src/main/java/
├── activations/
│   ├── ActivationFunction.java   # sigmoid, tanh, relu, leakyRelu, linear,
│   │                              # elu, selu, swish, gelu, softplus
│   └── Softmax.java              # softmax activation
├── data/
│   ├── MNISTDownloader.java      # generic gzip IDX downloader
│   ├── MNISTLoader.java          # generic IDX parser + resolve() + one-hot
│   ├── FashionMNISTDownloader.java
│   └── FashionMNISTLoader.java   # CLASS_NAMES, NUM_CLASSES, Dataset
├── evaluation/
│   ├── ClassificationMetrics.java # argmax, accuracy, confusion, P/R/F1
│   ├── HtmlReport.java            # self-contained HTML report
│   └── ImageEncoder.java          # float[] -> base64 PNG data URI
├── examples/
│   ├── FashionMNISTRunner.java   # main demo (auto-downloads)
│   ├── MNISTRunner.java
│   └── Main.java                 # dispatcher
├── initialization/
│   └── WeightInitializer.java    # xavierUniform, heNormal
├── layers/
│   ├── Neuron.java               # weights, bias, value, gradient
│   ├── Layer.java                # a row of neurons
│   ├── NeuralNetwork.java        # forward, loss, backprop, fit (SIMD)
│   ├── ModelFormat.java          # BINARY, JSON, CSV
│   ├── Dropout.java
│   ├── BatchNormalization.java
│   ├── Flatten.java
│   ├── Conv2D.java
│   └── MaxPool2D.java
├── losses/
│   ├── LossFunction.java         # MSE, BCE, softmax cross-entropy
│   ├── HingeLoss.java
│   ├── HuberLoss.java
│   ├── KLDivergenceLoss.java
│   ├── CategoricalCrossEntropyLoss.java
│   └── PoissonLoss.java
├── optim/
│   ├── Optimizer.java
│   ├── SGD.java
│   ├── Adam.java
│   ├── RMSProp.java
│   ├── Adagrad.java
│   └── Nadam.java
├── regularizers/
│   ├── L1Regularizer.java
│   ├── L2Regularizer.java
│   └── ElasticNetRegularizer.java
├── training/
│   └── Trainer.java              # epochs, batch size, shuffle, seed
└── vector/
    └── VectorMath.java           # SIMD dot, relu, addScaled (Vector API)

projects/
├── fashion-mnist/                # standalone Maven demo
└── mnist/                        # standalone Maven demo
```

---

[01 · Foundations →](01-foundations.md)
