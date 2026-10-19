package vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vectorized float operations using the JDK incubating Vector
 * API ({@code jdk.incubator.vector}). Accelerates the inner
 * loops of dense layer forward and backward passes.
 * Requires {@code --add-modules jdk.incubator.vector} at both
 * compile and run time.
 */
public final class VectorMath {

    /** Preferred species: 256-bit lanes (8 floats on AVX2). */
    private static final VectorSpecies<Float> SPECIES =
        FloatVector.SPECIES_PREFERRED;

    private VectorMath() {
    }

    /**
     * Dot product of two equal-length vectors using SIMD lanes.
     * Falls back to a scalar loop for the tail.
     *
     * @param a first vector
     * @param b second vector, same length as a
     * @return a . b
     * @throws IllegalArgumentException on null or length mismatch
     */
    public static float dot(float[] a, float[] b) {
        if (a == null || b == null || a.length != b.length) {
            throw new IllegalArgumentException(
                "Vectors must be non-null and of equal length");
        }

        int i = 0;
        float sum = 0.0f;
        int limit = SPECIES.loopBound(a.length);

        FloatVector acc = FloatVector.zero(SPECIES);

        for (; i < limit; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            acc = va.fma(vb, acc);            // acc += va * vb
        }

        sum = acc.reduceLanes(VectorOperators.ADD);

        for (; i < a.length; i++) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /**
     * ReLU over an array in place using SIMD max(0, x).
     *
     * @param x array modified in place
     * @throws IllegalArgumentException if null
     */
    public static void relu(float[] x) {
        if (x == null) {
            throw new IllegalArgumentException("Array required");
        }

        int i = 0;
        int limit = SPECIES.loopBound(x.length);
        FloatVector zero = FloatVector.zero(SPECIES);

        for (; i < limit; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(SPECIES, x, i);
            v.max(zero).intoArray(x, i);
        }

        for (; i < x.length; i++) {
            x[i] = Math.max(0.0f, x[i]);
        }
    }

    /**
     * Element-wise multiply-add: out[i] += a[i] * b[i] for the
     * first {@code n} elements. Used to accumulate weight
     * gradients without allocating temporaries.
     *
     * @param out accumulator, length &gt;= n
     * @param a   first factor
     * @param b   second factor
     * @param n   number of elements to process
     * @throws IllegalArgumentException on nulls or short arrays
     */
    public static void addScaled(float[] out, float[] a,
                                 float[] b, int n) {
        if (out == null || a == null || b == null ||
            out.length < n || a.length < n || b.length < n) {
            throw new IllegalArgumentException(
                "Arrays must be non-null and at least length n");
        }

        int i = 0;
        int limit = SPECIES.loopBound(n);

        for (; i < limit; i += SPECIES.length()) {
            FloatVector vo = FloatVector.fromArray(SPECIES, out, i);
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            vo = va.fma(vb, vo);              // vo += va * vb
            vo.intoArray(out, i);
        }

        for (; i < n; i++) {
            out[i] += a[i] * b[i];
        }
    }

    /**
     * Dot product of the first {@code n} elements of a and b.
     * Useful when the backing arrays are larger than the logical
     * vectors (e.g. a reusable buffer sized to the widest layer).
     *
     * @param a first vector, length &gt;= n
     * @param b second vector, length &gt;= n
     * @param n number of elements to use
     * @return a[0..n) . b[0..n)
     * @throws IllegalArgumentException on nulls or short arrays
     */
    public static float dot(float[] a, float[] b, int n) {
        if (a == null || b == null ||
            a.length < n || b.length < n) {
            throw new IllegalArgumentException(
                "Arrays must be non-null and at least length n");
        }

        int i = 0;
        int limit = SPECIES.loopBound(n);

        FloatVector acc = FloatVector.zero(SPECIES);

        for (; i < limit; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            acc = va.fma(vb, acc);
        }

        float sum = acc.reduceLanes(VectorOperators.ADD);

        for (; i < n; i++) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /**
     * Broadcast-scalar multiply-add with an offset into the
     * accumulator: {@code out[outOffset + i] += scalar * b[i]}
     * for {@code i} in [0, n). Used to scatter a neuron's
     * gradient into the flat parameter-gradient vector.
     *
     * @param out        accumulator array
     * @param outOffset  starting index in out
     * @param scalar     multiplier broadcast across lanes
     * @param b          source vector, length &gt;= n
     * @param n          number of elements to process
     * @throws IllegalArgumentException on nulls or short arrays
     */
    public static void addScaledScalar(float[] out, int outOffset,
                                       float scalar, float[] b,
                                       int n) {
        if (out == null || b == null ||
            out.length < outOffset + n || b.length < n) {
            throw new IllegalArgumentException(
                "Arrays must be non-null and long enough");
        }

        int i = 0;
        int limit = SPECIES.loopBound(n);
        FloatVector vScalar =
            FloatVector.broadcast(SPECIES, scalar);

        for (; i < limit; i += SPECIES.length()) {
            FloatVector vo = FloatVector.fromArray(
                SPECIES, out, outOffset + i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            vo = vScalar.fma(vb, vo);          // vo += scalar * vb
            vo.intoArray(out, outOffset + i);
        }

        for (; i < n; i++) {
            out[outOffset + i] += scalar * b[i];
        }
    }

    /**
     * Adds a scalar to every element in place.
     *
     * @param x     array modified in place
     * @param value scalar to add
     * @throws IllegalArgumentException if null
     */
    public static void addScalar(float[] x, float value) {
        if (x == null) {
            throw new IllegalArgumentException("Array required");
        }

        int i = 0;
        int limit = SPECIES.loopBound(x.length);
        FloatVector vScalar =
            FloatVector.broadcast(SPECIES, value);

        for (; i < limit; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(SPECIES, x, i);
            v.add(vScalar).intoArray(x, i);
        }

        for (; i < x.length; i++) {
            x[i] += value;
        }
    }
}
