/* Training-only fractional statistics, retaining NumPy's row reduction order. */
static PyObject *compact_statistics_kernel(PyObject *self, PyObject *args) {
    PyObject *o[9]; Py_buffer b[9] = {{0}};
    Py_ssize_t nr, width, ns; int min_states;
    double *restrict sum64 = NULL;
    float *restrict sum32 = NULL, *restrict mean32 = NULL, *restrict variance32 = NULL;
    uint8_t *restrict positive = NULL;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOnnni", &o[0], &o[1], &o[2], &o[3],
                          &o[4], &o[5], &o[6], &o[7], &o[8],
                          &nr, &width, &ns, &min_states)) return NULL;
    GET(0, 4, 0); GET(1, 1, 0); GET(2, 8, 0); GET(3, 8, 0);
    GET(4, 8, 1); GET(5, 1, 1); GET(6, 1, 1); GET(7, 8, 1); GET(8, 1, 1);
    REQUIRE(nr > 0 && width > 0 && ns > 0 && ns <= PY_SSIZE_T_MAX / 20);
    REQUIRE(shape(&b[0], nr, width, 4) && shape(&b[1], nr, ns, 1));
    REQUIRE(b[2].len / 8 == width && b[3].len / 8 == width);
    REQUIRE(b[4].len / 8 == ns && b[5].len == ns);
    REQUIRE(b[6].len == ns * 20 && shape(&b[7], ns, 20, 8) && b[8].len == ns * 20);
    REQUIRE(width <= PY_SSIZE_T_MAX / 8 && min_states >= 2 && min_states <= 20);
    const float *restrict raw = b[0].buf;
    const uint8_t *restrict observed = b[1].buf;
    const int64_t *restrict sites = b[2].buf, *restrict residues = b[3].buf;
    int64_t *restrict counts = b[4].buf;
    uint8_t *restrict eligible = b[5].buf, *restrict known = b[6].buf, *restrict active = b[8].buf;
    double *restrict means = b[7].buf;
    for (Py_ssize_t j = 0; j < width; ++j) {
        REQUIRE(sites[j] >= 0 && sites[j] < ns && residues[j] >= 0 && residues[j] < 20);
        REQUIRE(j == 0 || sites[j] > sites[j-1] ||
                (sites[j] == sites[j-1] && residues[j] > residues[j-1]));
    }
    sum64 = PyMem_Calloc((size_t)width, sizeof(double));
    sum32 = PyMem_Calloc((size_t)width, sizeof(float));
    mean32 = PyMem_Calloc((size_t)width, sizeof(float));
    variance32 = PyMem_Calloc((size_t)width, sizeof(float));
    positive = PyMem_Calloc((size_t)width, sizeof(uint8_t));
    if (!sum64 || !sum32 || !mean32 || !variance32 || !positive) { PyErr_NoMemory(); goto fail; }
    memset(counts, 0, (size_t)ns * 8); memset(eligible, 0, (size_t)ns);
    memset(known, 0, (size_t)ns * 20); memset(means, 0, (size_t)ns * 20 * 8);
    memset(active, 0, (size_t)ns * 20);
    int valid = 1;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nr; ++i) {
        for (Py_ssize_t s = 0; s < ns; ++s) counts[s] += observed[i * ns + s] != 0;
        for (Py_ssize_t j = 0; j < width; ++j) {
            float q = raw[i * width + j];
            if (isnan(q)) q = 0;
            else if (!isfinite(q) || q < 0 || q > 1) { valid = 0; q = 0; }
            positive[j] |= q > 0;
            sum64[j] += (double)q;
            sum32[j] = sum32[j] + q;
        }
    }
    for (Py_ssize_t j = 0; j < width; ++j)
        known[sites[j] * 20 + residues[j]] = positive[j];
    for (Py_ssize_t s = 0; s < ns; ++s) {
        int states = 0;
        for (int aa = 0; aa < 20; ++aa) states += known[s * 20 + aa] != 0;
        eligible[s] = states >= min_states;
    }
    /* Eligible sites have at least two columns. NumPy reduces a C-order
       (samples, selected_states) array down its strided axis, sequentially.
       float32/int64 divide uses float64 arithmetic before casting the output. */
    for (Py_ssize_t j = 0; j < width; ++j) if (eligible[sites[j]] && counts[sites[j]]) {
        double denominator = (double)counts[sites[j]];
        means[sites[j] * 20 + residues[j]] = sum64[j] / denominator;
        mean32[j] = (float)((double)sum32[j] / denominator);
    }
    /* Keep contiguous accumulators even for ineligible columns; their outputs
       remain zero. Avoiding a per-column eligibility branch permits SIMD while
       every eligible column retains its original training-row addition order. */
    if (valid) for (Py_ssize_t i = 0; i < nr; ++i) {
        for (Py_ssize_t j = 0; j < width; ++j) {
            float q = raw[i * width + j];
            if (isnan(q)) q = 0;
            float centered = q - mean32[j];
            if (!observed[i * ns + sites[j]]) centered = 0;
            float square = centered * centered;
            variance32[j] = variance32[j] + square;
        }
    }
    for (Py_ssize_t j = 0; j < width; ++j) if (eligible[sites[j]] && counts[sites[j]]) {
        float variance = (float)((double)variance32[j] / (double)counts[sites[j]]);
        active[sites[j] * 20 + residues[j]] = variance > 1e-12f;
    }
    Py_END_ALLOW_THREADS
    PyMem_Free(sum64); PyMem_Free(sum32); PyMem_Free(mean32); PyMem_Free(variance32);
    PyMem_Free(positive);
    release(b, 9);
    return PyBool_FromLong(valid);
fail:
    PyMem_Free(sum64); PyMem_Free(sum32); PyMem_Free(mean32); PyMem_Free(variance32);
    PyMem_Free(positive);
    release(b, 9); return NULL;
}
