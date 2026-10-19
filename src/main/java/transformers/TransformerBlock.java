package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * A single transformer block consisting of:
 * - Multi-head self-attention with pre-normalisation (LayerNorm)
 * - Add & norm
 * - Position-wise feed-forward
 * - Add & norm
 */
public final class TransformerBlock {

    private final LayerNorm attnNorm;
    private final MultiHeadAttention attention;
    private final LayerNorm ffnNorm;
    private final FeedForward ffn;
    private final int dModel;

    public TransformerBlock(int dModel, int numHeads,
                            int ffnHidden, long seed) {
        if (dModel <= 0 || numHeads <= 0 || ffnHidden <= 0) {
            throw new IllegalArgumentException(
                "all dimensions must be positive");
        }
        this.dModel = dModel;
        this.attnNorm = new LayerNorm(dModel);
        this.attention = new MultiHeadAttention(
            dModel, numHeads, seed);
        this.ffnNorm = new LayerNorm(dModel);
        this.ffn = new FeedForward(dModel, ffnHidden, seed + 5);
    }

    /**
     * Applies the transformer block.
     *
     * @param x         input of shape {B, T, D}
     * @param useCausal apply causal mask (true for decoder)
     * @return output of shape {B, T, D}
     */
    public Tensor forward(Tensor x, boolean useCausal) {
        // Pre-norm, residual
        Tensor h = attnNorm.forward(x);
        h = attention.forward(h, useCausal);
        h = Ops.add(h, x);

        // Feed-forward
        Tensor h2 = ffnNorm.forward(h);
        h2 = ffn.forward(h2);
        return Ops.add(h2, h);
    }

    public Tensor forward(Tensor x) {
        return forward(x, false);
    }

    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(attnNorm.parameters());
        p.addAll(attention.parameters());
        p.addAll(ffnNorm.parameters());
        p.addAll(ffn.parameters());
        return p;
    }
}
