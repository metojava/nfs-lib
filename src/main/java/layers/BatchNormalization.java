package layers;

/**
 * Batch normalization: normalizes each feature across the batch
 * to zero mean and unit variance, then applies a learned affine
 * transform (scale gamma, shift beta). Reduces internal
 * covariate shift and allows higher learning rates.
 * Works on a full mini-batch at once.
 */
public class BatchNormalization {

    private final float gamma;
    private final float beta;
    private final float epsilon;
    private final float momentum;

    /** Running estimates for inference. */
    private float runningMean;
    private float runningVar;
    private boolean initialized;

    public BatchNormalization(float gamma, float beta,
                              float epsilon, float momentum) {
        this.gamma = gamma;
        this.beta = beta;
        this.epsilon = epsilon;
        this.momentum = momentum;
        this.runningMean = 0.0f;
        this.runningVar = 1.0f;
        this.initialized = false;
    }

    public static BatchNormalization of() {
        return new BatchNormalization(1.0f, 0.0f,
            1e-5f, 0.1f);
    }

    /**
     * Normalizes a batch in place using batch statistics, and
     * updates the running estimates for later inference use.
     *
     * @param batch [N][features], each row modified in place
     */
    public void normalize(float[][] batch) {
        if (batch == null || batch.length == 0) {
            throw new IllegalArgumentException(
                "Batch must be nonempty");
        }

        int n = batch.length;
        int features = batch[0].length;

        for (int f = 0; f < features; f++) {
            float mean = 0.0f;
            for (int i = 0; i < n; i++) {
                mean += batch[i][f];
            }
            mean /= n;

            float variance = 0.0f;
            for (int i = 0; i < n; i++) {
                float d = batch[i][f] - mean;
                variance += d * d;
            }
            variance /= n;

            float invStd =
                1.0f / (float) Math.sqrt(variance + epsilon);

            for (int i = 0; i < n; i++) {
                batch[i][f] = gamma
                    * (batch[i][f] - mean) * invStd + beta;
            }

            if (!initialized) {
                runningMean = mean;
                runningVar = variance;
                initialized = true;
            } else {
                runningMean = momentum * runningMean
                            + (1.0f - momentum) * mean;
                runningVar  = momentum * runningVar
                            + (1.0f - momentum) * variance;
            }
        }
    }

    /**
     * Normalizes a single vector using the stored running
     * statistics (for inference after training).
     *
     * @param x one example, modified in place
     */
    public void normalizeInference(float[] x) {
        for (int f = 0; f < x.length; f++) {
            float invStd = 1.0f / (float) Math.sqrt(
                runningVar + epsilon);
            x[f] = gamma * (x[f] - runningMean) * invStd + beta;
        }
    }
}
