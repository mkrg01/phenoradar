/* Training-only fractional statistics, retaining NumPy's row reduction order. */
static PyObject *compact_statistics_kernel(PyObject *self, PyObject *args) {
    PyObject *o[9]; Py_buffer b[9] = {{0}};
    Py_ssize_t nr, width, ns; int min_states;
    double *restrict sum64 = NULL;
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
    positive = PyMem_Calloc((size_t)width, sizeof(uint8_t));
    if (!sum64 || !positive) { PyErr_NoMemory(); goto fail; }
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
        }
    }
    for (Py_ssize_t j = 0; j < width; ++j)
        known[sites[j] * 20 + residues[j]] = positive[j];
    for (Py_ssize_t s = 0; s < ns; ++s) {
        int states = 0;
        for (int aa = 0; aa < 20; ++aa) states += known[s * 20 + aa] != 0;
        eligible[s] = states >= min_states;
    }
    /* Keep every training-observed state; the solver handles constant columns. */
    for (Py_ssize_t j = 0; j < width; ++j) if (eligible[sites[j]]) {
        Py_ssize_t index = sites[j] * 20 + residues[j];
        active[index] = positive[j];
        if (counts[sites[j]]) means[index] = sum64[j] / (double)counts[sites[j]];
    }
    Py_END_ALLOW_THREADS
    PyMem_Free(sum64);
    PyMem_Free(positive);
    release(b, 9);
    return PyBool_FromLong(valid);
fail:
    PyMem_Free(sum64);
    PyMem_Free(positive);
    release(b, 9); return NULL;
}

/* Count minor support directly from the stored residue columns. */
static PyObject *compact_minor_support(PyObject *self, PyObject *args) {
    PyObject *o[7]; Py_buffer b[7] = {{0}};
    uint64_t *seen = NULL;
    if (!PyArg_UnpackTuple(args, "compact_minor_support", 7, 7,
                          &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6])) return NULL;
    GET(0, 4, 0); GET(1, 8, 0); GET(2, 8, 0); GET(3, 8, 0);
    GET(4, 8, 1); GET(5, 8, 1); GET(6, 1, 1);
    Py_ssize_t nr = b[3].len / 8, ns = b[2].len / 8;
    REQUIRE(nr > 0 && ns > 0 && b[0].len % (4 * nr) == 0);
    Py_ssize_t width = b[0].len / 4 / nr;
    REQUIRE(b[1].len / 8 == ns + 1 && b[4].len / 8 == ns && b[5].len / 8 == ns && b[6].len == ns);
    const float *raw = b[0].buf;
    const int64_t *offsets = b[1].buf, *major = b[2].buf, *groups = b[3].buf;
    int64_t *sample_counts = b[4].buf, *group_counts = b[5].buf;
    uint8_t *fallback = b[6].buf;
    REQUIRE(offsets[0] == 0 && offsets[ns] == width);
    for (Py_ssize_t s = 0; s < ns; ++s) {
        REQUIRE(offsets[s + 1] >= offsets[s] && offsets[s + 1] - offsets[s] <= 20);
        REQUIRE(major[s] == -1 || (major[s] >= offsets[s] && major[s] < offsets[s + 1]));
    }
    for (Py_ssize_t i = 0; i < nr; ++i) { REQUIRE(groups[i] >= -1 && groups[i] < 64); }
    seen = PyMem_Calloc((size_t)ns, sizeof(uint64_t));
    if (!seen) { PyErr_NoMemory(); goto fail; }
    int valid = 1;
    const double threshold = (double)1e-7f;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nr && valid; ++i) {
        const float *row = raw + i * width;
        uint64_t group = groups[i] < 0 ? 0 : UINT64_C(1) << groups[i];
        for (Py_ssize_t s = 0; s < ns; ++s) {
            if (fallback[s]) continue;
            double total = 0, minor = 0;
            for (int64_t j = offsets[s]; j < offsets[s + 1]; ++j) {
                float q = row[j];
                if (isnan(q)) continue;
                if (q < 0 || q > 1) { valid = 0; break; }
                total += q;
                if (j != major[s]) minor += q;
            }
            if (minor == 0) continue;
            /* Bound any float32 reduction of 20 nonnegative values and its
               subtraction. Close calls use the original NumPy expression. */
            if (fabs(minor - threshold) <= 32 * FLT_EPSILON * total) {
                fallback[s] = 1;
            } else if (minor > threshold) {
                sample_counts[s]++;
                group_counts[s] += (group & ~seen[s]) != 0;
                seen[s] |= group;
            }
        }
    }
    Py_END_ALLOW_THREADS
    PyMem_Free(seen);
    release(b, 7);
    return PyBool_FromLong(valid);
fail:
    release(b, 7);
    return NULL;
}
