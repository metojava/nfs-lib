package layers;

import java.util.Random;

/**
 * 2-D convolution layer (single-channel input, single or
 * multi-channel output). Operates on float[H][W] feature maps
 * with a float[kh][kw] kernel per output channel. Stride and
 * zero-padding are configurable. Weights are initialized with
 * He normal for ReLU compatibility.
 */
public class Conv2D {

    /** Kernel weights: [outChannels][kh][kw]. */
    public final float[][][] weights;

    /** Biases: [outChannels]. */
    public final float[] biases;

    private final int outChannels;
    private final int kernelH;
    private final int kernelW;
    private final int stride;
    private final int padding;

    /**
     * Creates a Conv2D layer.
     *
     * @param outChannels number of output filters
     * @param kernelH     kernel height
     * @param kernelW     kernel width
     * @param stride      step size for the sliding window
     * @param padding     zeros added on each border
     * @param seed        RNG seed for weight init
     */
    public Conv2D(int outChannels, int kernelH, int kernelW,
                  int stride, int padding, long seed) {
        if (outChannels <= 0 || kernelH <= 0 || kernelW <= 0
            || stride <= 0 || padding < 0) {
            throw new IllegalArgumentException(
                "Invalid Conv2D parameters");
        }

        this.outChannels = outChannels;
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.stride = stride;
        this.padding = padding;
        this.weights = new float[outChannels][kernelH][kernelW];
        this.biases = new float[outChannels];

        Random rand = new Random(seed);
        float scale = (float) Math.sqrt(
            2.0f / (kernelH * kernelW));

        for (int oc = 0; oc < outChannels; oc++) {
            for (int i = 0; i < kernelH; i++) {
                for (int j = 0; j < kernelW; j++) {
                    weights[oc][i][j] =
                        (float) rand.nextGaussian() * scale;
                }
            }
        }
    }

    /**
     * Output height for a given input height.
     */
    public int outputSize(int inputSize) {
        return (inputSize + 2 * padding - kernelH) / stride + 1;
    }

    /**
     * Convolves a single-channel input map and returns one
     * output map per filter.
     *
     * @param input [height][width] feature map
     * @return [outChannels][outH][outW] feature maps
     */
    public float[][][] forward(float[][] input) {
        int inH = input.length;
        int inW = input[0].length;
        int outH = outputSize(inH);
        int outW = outputSize(inW);

        float[][][] output =
            new float[outChannels][outH][outW];

        for (int oc = 0; oc < outChannels; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = biases[oc];

                    for (int kh = 0; kh < kernelH; kh++) {
                        for (int kw = 0; kw < kernelW; kw++) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;

                            if (ih >= 0 && ih < inH &&
                                iw >= 0 && iw < inW) {
                                sum += input[ih][iw]
                                     * weights[oc][kh][kw];
                            }
                        }
                    }

                    output[oc][oh][ow] = sum;
                }
            }
        }

        return output;
    }

    /** Number of output channels. */
    public int outChannels() {
        return outChannels;
    }
}
