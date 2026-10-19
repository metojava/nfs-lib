package optim;

/**
 * Adam: adaptive moment estimation. Maintains an exponential
 * moving average of the gradient (first moment) and of the
 * squared gradient (second moment), applies bias correction,
 * and divides the mean by the root mean square.
 * Defaults: beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8.
 */
public class Adam implements Optimizer {

    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;

    private float[] firstMoment;
    private float[] secondMoment;
    private long timestep;

    /**
     * Adam with the standard hyper-parameters.
     *
     * @param learningRate step size, must be &gt; 0
     */
    public Adam(float learningRate) {
        this(learningRate, 0.9f, 0.999f, 1e-8f);
    }

    /**
     * Adam with explicit hyper-parameters.
     *
     * @param learningRate step size, must be &gt; 0
     * @param beta1        first-moment decay in [0, 1)
     * @param beta2        second-moment decay in [0, 1)
     * @param epsilon      denominator guard, must be &gt; 0
     * @throws IllegalArgumentException on invalid parameters
     */
    public Adam(float learningRate, float beta1,
                float beta2, float epsilon) {
        if (!(learningRate > 0.0f) ||
            beta1 < 0.0f || beta1 >= 1.0f ||
            beta2 < 0.0f || beta2 >= 1.0f ||
            !(epsilon > 0.0f)) {
            throw new IllegalArgumentException(
                "Invalid Adam parameters");
        }

        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    /**
     * One Adam step with bias-corrected moments:
     * m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g^2;
     * w -= lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps).
     *
     * @param parameters weights, updated in place
     * @param gradients  their gradients; same length required
     * @throws IllegalArgumentException on length mismatch
     */
    @Override
    public void update(float[] parameters, float[] gradients) {
        if (parameters.length != gradients.length) {
            throw new IllegalArgumentException(
                "Parameter/gradient mismatch");
        }

        if (firstMoment == null ||
            firstMoment.length != parameters.length) {
            firstMoment = new float[parameters.length];
            secondMoment = new float[parameters.length];
            timestep = 0;
        }

        timestep++;
        float correction1 =
            1.0f - (float) Math.pow(beta1, timestep);
        float correction2 =
            1.0f - (float) Math.pow(beta2, timestep);

        for (int i = 0; i < parameters.length; i++) {
            float g = gradients[i];

            firstMoment[i] = beta1 * firstMoment[i]
                           + (1.0f - beta1) * g;

            secondMoment[i] = beta2 * secondMoment[i]
                            + (1.0f - beta2) * g * g;

            float mHat = firstMoment[i] / correction1;
            float vHat = secondMoment[i] / correction2;

            parameters[i] -= learningRate * mHat
                           / ((float) Math.sqrt(vHat) + epsilon);
        }
    }
}
