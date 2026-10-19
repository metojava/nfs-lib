package activations;

/**
 * Swish (also called SiLU): x * sigmoid(beta * x). Smooth,
 * non-monotonic, often outperforms ReLU on deeper networks.
 * With beta = 1 this is the SiLU activation.
 */
public final class Swish implements ActivationFunction {

    private final float beta;

    public Swish(float beta) {
        if (beta <= 0.0f) {
            throw new IllegalArgumentException(
                "beta must be positive");
        }
        this.beta = beta;
    }

    public static Swish of() {
        return new Swish(1.0f);
    }

    public static Swish of(float beta) {
        return new Swish(beta);
    }

    @Override
    public float activate(float x) {
        float s = sigmoid(beta * x);
        return x * s;
    }

    @Override
    public float derivativeFromOutput(float y) {
        // y = x * sigmoid(beta*x)
        // f' = sigmoid(beta*x) + x * beta * sigmoid * (1 - sigmoid)
        // Recover sigmoid from y/x is fragile; recompute from
        // the relation s = y/x only when x != 0. For the common
        // case we compute directly.
        throw new UnsupportedOperationException(
            "Use derivativeFromInput for Swish");
    }

    /**
     * Derivative expressed in terms of the pre-activation x.
     *
     * @param x pre-activation input
     * @return f'(x)
     */
    public float derivativeFromInput(float x) {
        float s = sigmoid(beta * x);
        return s + x * beta * s * (1.0f - s);
    }

    private static float sigmoid(float x) {
        if (x >= 0.0f) {
            return 1.0f / (1.0f + (float) Math.exp(-x));
        }
        float e = (float) Math.exp(x);
        return e / (1.0f + e);
    }
}
