/* Decode immutable cache fields directly into species-major output buffers. */
static int buffers_overlap(const Py_buffer *a, const Py_buffer *b) {
    if (!a->len || !b->len) return 0;
    uintptr_t x = (uintptr_t)a->buf, y = (uintptr_t)b->buf;
    return x <= y ? y - x < (uintptr_t)a->len : x - y < (uintptr_t)b->len;
}

static PyObject *decode_compact(PyObject *self, PyObject *args) {
    PyObject *data_object, *axes_object, *outputs_object, *axes = NULL, *outputs = NULL;
    Py_buffer b[10] = {{0}};
    Py_ssize_t nr, width;
    uint64_t *starts = NULL, *prefix = NULL;
    int packed;
    if (!PyArg_ParseTuple(args, "OnnOOp", &data_object, &nr, &width,
                          &axes_object, &outputs_object, &packed)) return NULL;
    axes = PySequence_Fast(axes_object, "Expected compact decoding axes");
    if (!axes) goto fail;
    outputs = PySequence_Fast(outputs_object, "Expected compact output buffers");
    if (!outputs) goto fail;
    REQUIRE(PySequence_Fast_GET_SIZE(axes) == 6 && PySequence_Fast_GET_SIZE(outputs) == 3);
    if (!get(data_object, &b[0], 1, 0)) goto fail;
    for (int j = 0; j < 6; ++j)
        if (!get(PySequence_Fast_GET_ITEM(axes, j), &b[j + 1], 8, 0)) goto fail;
    for (int j = 0; j < 3; ++j) {
        PyObject *output = PySequence_Fast_GET_ITEM(outputs, j);
        if (j == 2 && output == Py_None) continue;
        if (!get(output, &b[j + 7], j == 1 ? 1 : 4, 1)) goto fail;
    }
    REQUIRE(nr >= 0 && width >= 0 && width < PY_SSIZE_T_MAX - 7);
    REQUIRE(b[7].ndim == 2 && b[8].ndim == 2);
    Py_ssize_t nt = b[3].len / 8, nc = b[1].len / 8, nf = b[2].len / 8;
    Py_ssize_t n = b[7].shape[0], features_width = b[7].shape[1], sites_width = b[8].shape[1];
    REQUIRE(b[1].len == nc * 8 && b[2].len == nf * 8 && b[3].len == nt * 8);
    REQUIRE(b[4].len == nt * 8 && b[5].len == nc * 8 && b[6].len == nf * 8);
    REQUIRE(b[8].shape[0] == n && shape(&b[7], n, features_width, 4));
    REQUIRE(shape(&b[8], n, sites_width, 1));
    REQUIRE(!b[9].obj || shape(&b[9], n, sites_width, 4));
    REQUIRE(nf <= PY_SSIZE_T_MAX - nc);
    for (int j = 7; j < 10; ++j)
        if (b[j].obj)
            for (int k = 0; k < j; ++k)
                if (b[k].obj) REQUIRE(!buffers_overlap(&b[j], &b[k]));
    const int64_t *coverage_rows = b[1].buf, *state_rows = b[2].buf;
    const int64_t *samples = b[3].buf, *targets = b[4].buf;
    const int64_t *selected = b[5].buf, *features = b[6].buf;
    for (Py_ssize_t j = 0; j < nt; ++j)
        REQUIRE(samples[j] >= 0 && samples[j] < width && targets[j] >= 0 && targets[j] < n);
    for (Py_ssize_t j = 0; j < nc; ++j)
        REQUIRE(coverage_rows[j] >= 0 && coverage_rows[j] < nr
                && selected[j] >= 0 && selected[j] < sites_width);
    for (Py_ssize_t j = 0; j < nf; ++j)
        REQUIRE(state_rows[j] >= 0 && state_rows[j] < nr
                && features[j] >= 0 && features[j] < features_width);
    Py_ssize_t nb = (width + 7) / 8, end = 0;
    if (packed) {
        REQUIRE(nb == 0 || nr <= PY_SSIZE_T_MAX / 3 / nb);
        end = nr * 3 * nb;
        REQUIRE(end <= b[0].len && nr < PY_SSIZE_T_MAX);
        starts = PyMem_Calloc((size_t)nr + 1, sizeof(uint64_t));
        prefix = PyMem_Calloc((size_t)nb + 1, sizeof(uint64_t));
        if (!starts || !prefix) { PyErr_NoMemory(); goto fail; }
    } else REQUIRE(shape(&b[0], nr, width, 4));
    const uint8_t *data = b[0].buf;
    uint32_t *values = b[7].buf, *coverage = b[9].buf;
    uint8_t *observed = b[8].buf;
    int invalid = 0;
    Py_BEGIN_ALLOW_THREADS
    if (packed) {
        for (Py_ssize_t row = 0; row < nr; ++row) {
            starts[row + 1] = starts[row];
            const uint8_t *exceptions = data + (row * 3 + 2) * nb;
            for (Py_ssize_t j = 0; j < nb; ++j) starts[row + 1] += pop(exceptions[j]);
        }
        invalid = starts[nr] > (uint64_t)((b[0].len - end) / 4);
    }
    if (!invalid) for (Py_ssize_t f = 0; f < nc + nf; ++f) {
        int is_coverage = f < nc;
        Py_ssize_t row = is_coverage ? coverage_rows[f] : state_rows[f - nc];
        Py_ssize_t target = is_coverage ? selected[f] : features[f - nc];
        const uint8_t *planes = packed ? data + row * 3 * nb : NULL;
        if (packed) {
            prefix[0] = 0;
            for (Py_ssize_t j = 0; j < nb; ++j)
                prefix[j + 1] = prefix[j] + pop(planes[2 * nb + j]);
        }
        for (Py_ssize_t i = 0; i < nt; ++i) {
            uint32_t value;
            if (!packed) memcpy(&value, data + (row * width + samples[i]) * 4, 4);
            else {
                Py_ssize_t j = samples[i] / 8;
                unsigned bit = 1U << (samples[i] % 8);
                value = planes[j] & bit ? 0x3f800000 : 0;
                if (planes[nb + j] & bit) value = 0x7fc00000;
                if (planes[2 * nb + j] & bit) {
                    uint64_t index = starts[row] + prefix[j] + pop(planes[2 * nb + j] & (bit - 1));
                    memcpy(&value, data + end + index * 4, 4);
                }
            }
            if (is_coverage) {
                Py_ssize_t offset = targets[i] * sites_width + target;
                int finite = (value & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
                observed[offset] = (uint8_t)finite;
                if (coverage) coverage[offset] = finite ? value : 0;
            } else values[targets[i] * features_width + target] = value;
        }
    }
    Py_END_ALLOW_THREADS
    if (invalid) { PyErr_SetString(PyExc_ValueError, "Truncated packed frequencies"); goto fail; }
    PyMem_Free(starts); PyMem_Free(prefix); release(b, 10);
    Py_DECREF(axes); Py_DECREF(outputs); Py_RETURN_NONE;
fail:
    PyMem_Free(starts); PyMem_Free(prefix); release(b, 10);
    Py_XDECREF(axes); Py_XDECREF(outputs); return NULL;
}
