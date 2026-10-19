package transformers;

import autograd.Ops;
import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Multi-head scaled dot-product attention.
 *
 * <p>For input of shape {B, T, D}, we project to queries,
 * keys, values, split into H heads, compute attention, then
 * concatenate back to {B, T, D}.</p>
 */
public final class MultiHeadAttention {

    private final Linear qProj;
    private final Linear kProj;
    private final Linear vProj;
    private final Linear outProj;
    private final int dModel;
    private final int numHeads;
    private final int headDim;

    public MultiHeadAttention(int dModel, int numHeads,
                              long seed) {
        if (dModel <= 0 || numHeads <= 0
                || dModel % numHeads != 0) {
            throw new IllegalArgumentException(
                "dModel must be divisible by numHeads, got dModel="
                + dModel + ", numHeads=" + numHeads);
        }
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;
        this.qProj = new Linear(dModel, dModel, seed + 1);
        this.kProj = new Linear(dModel, dModel, seed + 2);
        this.vProj = new Linear(dModel, dModel, seed + 3);
        this.outProj = new Linear(dModel, dModel, seed + 4);
    }

    /**
     * Forward pass with causal mask (for GPT) or optional mask.
     *
     * @param x        input of shape {B, T, D}
     * @param useCausal mask future tokens
     * @return attention output of shape {B, T, D}
     */
    public Tensor forward(Tensor x, boolean useCausal) {
        if (x.ndim() != 3) {
            throw new IllegalArgumentException(
                "expected 3D {B, T, D}, got " + x.ndim() + "D");
        }
        if (x.size(2) != dModel) {
            throw new IllegalArgumentException(
                "expected D=" + dModel + ", got " + x.size(2));
        }

        int B = x.size(0), T = x.size(1);
        Tensor Q = qProj.forward(x); // {B, T, D}
        Tensor K = kProj.forward(x); // {B, T, D}
        Tensor V = vProj.forward(x); // {B, T, D}

        Q = reshapeHeads(Q, B, T);
        K = reshapeHeads(K, B, T);
        V = reshapeHeads(V, B, T); // {B, H, T, dH}

        // Q @ K^T / sqrt(dH)
        Tensor KT = Ops.transpose(K, 2, 3); // {B, H, dH, T}
        Tensor scores = Ops.matmul(Q, KT);
        float scale = 1.0f / (float) Math.sqrt(headDim);
        scores = Ops.scale(scores, scale);

        if (useCausal) {
            Tensor mask = Ops.causalMask(T);
            scores = Ops.add(scores, mask);
        }

        Tensor attn = Ops.softmax(scores); // {B, H, T, T}
        Tensor outHeads = Ops.matmul(attn, V); // {B, H, T, dH}
        Tensor out = unreshapeHeads(outHeads, B, T, dModel);
        return outProj.forward(out);
    }

    /**
     * Convenience for encoder (self-attention, no causal mask).
     */
    public Tensor forward(Tensor x) {
        return forward(x, false);
    }

    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(qProj.parameters());
        p.addAll(kProj.parameters());
        p.addAll(vProj.parameters());
        p.addAll(outProj.parameters());
        return p;
    }

    private Tensor reshapeHeads(Tensor x, int B, int T) {
        Tensor flat = Ops.reshape(x, B, T, numHeads, headDim);
        return Ops.transpose(flat, 1, 2); // {B, H, T, dH}
    }

    private Tensor unreshapeHeads(Tensor x, int B, int T,
                                  int dModel) {
        Tensor tr = Ops.transpose(x, 1, 2); // {B, T, H, dH}
        return Ops.reshape(tr, B, T, dModel);
    }
}
