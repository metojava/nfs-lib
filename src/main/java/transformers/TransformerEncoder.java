package transformers;

import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Transformer encoder (BERT-style): stack of TransformerBlock
 * layers with no causal masking. Used for bidirectional
 * sequence encoding.
 */
public final class TransformerEncoder {

    private final Embedding embedding;
    private final List<TransformerBlock> blocks;
    private final LayerNorm finalNorm;
    private final int numLayers;

    public TransformerEncoder(int vocabSize, int dModel,
                              int numHeads, int numLayers,
                              int ffnHidden, long seed) {
        if (vocabSize <= 0 || dModel <= 0 || numHeads <= 0
                || numLayers <= 0 || ffnHidden <= 0) {
            throw new IllegalArgumentException(
                "all parameters must be positive");
        }
        this.numLayers = numLayers;
        this.embedding = new Embedding(vocabSize, dModel,
            seed + 100);
        this.blocks = new ArrayList<>(numLayers);
        for (int i = 0; i < numLayers; i++) {
            long s = seed + 200 + i * 10;
            blocks.add(new TransformerBlock(dModel, numHeads,
                ffnHidden, s));
        }
        this.finalNorm = new LayerNorm(dModel);
    }

    /**
     * Encodes a batch of token sequences.
     *
     * @param ids       token ids of length B*T
     * @param batchSize batch size
     * @param seqLen    sequence length
     * @return encoder hidden states of shape {B, T, dModel}
     */
    public Tensor forward(int[] ids, int batchSize,
                          int seqLen) {
        Tensor x = embedding.forward(ids, batchSize, seqLen);
        for (int i = 0; i < numLayers; i++) {
            x = blocks.get(i).forward(x, false);
        }
        return finalNorm.forward(x);
    }

    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(embedding.parameters());
        for (TransformerBlock b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(finalNorm.parameters());
        return p;
    }
}
