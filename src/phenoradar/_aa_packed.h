/* Included by _aa_kernels.c. Code zero denotes a structural zero. */
static int first_bit(uint64_t nonzero) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(nonzero);
#else
    int bit = 0;
    while (!(nonzero & 1)) { nonzero >>= 1; ++bit; }
    return bit;
#endif
}

static uint64_t bits(double value) {
    uint64_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

/* Return the number of packed columns. Keep exceptions in ordinary CSC. */
static int32_t packed_pack(int32_t width, int32_t words, const double *data,
             const int32_t *rows, const int32_t *ptr, int32_t *mapping,
             double *values, uint64_t *planes) {
    int32_t packed = 0, fallback = 0;
    for (int32_t col = 0; col < width; col++) {
        int32_t start = ptr[col], stop = ptr[col + 1];
        double palette[3] = {0, 0, 0};
        uint64_t palette_bits[3];
        int count = 0;
        /* Packing must save space even after accounting for the palette. */
        if (12 * (int64_t)(stop - start) > 16 * (int64_t)words + 24) {
            for (int32_t i = start; i < stop; i++) {
                uint64_t value = bits(data[i]);
                int code = 0;
                while (code < count && palette_bits[code] != value) code++;
                if (code == 3) {
                    count = 4;
                    break;
                }
                if (code == count) {
                    palette[count] = data[i];
                    palette_bits[count++] = value;
                }
            }
            if (count <= 3) {
                uint64_t *low = planes + (int64_t)packed * 2 * words;
                uint64_t *high = low + words;
                memset(low, 0, 2 * (size_t)words * sizeof(*low));
                memcpy(values + 3 * (int64_t)packed, palette, sizeof(palette));
                for (int32_t i = start; i < stop; i++) {
                    int code = 0;
                    uint64_t value = bits(data[i]);
                    while (palette_bits[code] != value) code++;
                    int32_t word = rows[i] / 64;
                    uint64_t bit = UINT64_C(1) << (rows[i] % 64);
                    if ((code + 1) & 1) low[word] |= bit;
                    if ((code + 1) & 2) high[word] |= bit;
                }
                mapping[col] = packed++;
                continue;
            }
        }
        mapping[col] = -1 - fallback++;
    }
    return packed;
}

/* Iterate over nonzeros in exactly the original CSC row order. */
static inline void scan_impl(int fused, int32_t width, int32_t words, const int32_t *mapping,
          const double *values, const uint64_t *planes,
          const double *data, const int32_t *rows, const int32_t *ptr,
          const double *residual, double *out) {
    for (int32_t col = 0; col < width; col++) {
        int32_t index = mapping[col];
        double total = 0;
        if (index < 0) {
            int32_t fallback = -1 - index;
            for (int32_t i = ptr[fallback]; i < ptr[fallback + 1]; i++)
                total = fused ? fma(data[i], residual[rows[i]], total)
                              : total + data[i] * residual[rows[i]];
        } else {
            const double *palette = values + 3 * (int64_t)index;
            const uint64_t *low = planes + (int64_t)index * 2 * words;
            const uint64_t *high = low + words;
            for (int32_t word = 0; word < words; word++) {
                uint64_t active = low[word] | high[word];
                while (active) {
                    int bit = first_bit(active);
                    int code = ((low[word] >> bit) & 1) + 2 * ((high[word] >> bit) & 1);
                    double r = residual[64 * word + bit];
                    total = fused ? fma(palette[code - 1], r, total)
                                  : total + palette[code - 1] * r;
                    active &= active - 1;
                }
            }
        }
        out[col] = total;
    }
}

static void packed_take(int32_t words, const double *palette, const uint64_t *low,
          double *data, int32_t *rows) {
    const uint64_t *high = low + words;
    for (int32_t word = 0; word < words; word++) {
        uint64_t active = low[word] | high[word];
        while (active) {
            int bit = first_bit(active);
            int code = ((low[word] >> bit) & 1) + 2 * ((high[word] >> bit) & 1);
            *data++ = palette[code - 1];
            *rows++ = 64 * word + bit;
            active &= active - 1;
        }
    }
}

/* Encode binary-cache rows directly into the packed gradient representation. */
static void packed_pack_binary(int32_t width, int32_t height, int32_t stride,
                 const uint8_t *ones, const uint8_t *mass, const int64_t *sites,
                 const int64_t *sources, const double *palette, uint64_t *planes,
                 int32_t *counts) {
    int32_t words = height / 64 + (height % 64 != 0);
    for (int32_t col = 0; col < width; col++) {
        uint64_t *low = planes + (int64_t)col * 2 * words;
        uint64_t *high = low + words;
        const uint8_t *q = ones + (int64_t)col * stride;
        const uint8_t *m = mass + sites[col] * stride;
        int32_t count = 0;
        for (int32_t row = 0; row < height; row++) {
            int64_t source = sources[row];
            int category = 1;
            if (source >= 0) {
                unsigned bit = 1u << (source % 8);
                if (q[source / 8] & bit) category = 2;
                else if (m[source / 8] & bit) category = 0;
            }
            if (palette[3 * (int64_t)col + category] != 0.) {
                uint64_t bit = UINT64_C(1) << (row % 64);
                if ((category + 1) & 1) low[row / 64] |= bit;
                if ((category + 1) & 2) high[row / 64] |= bit;
                count++;
            }
        }
        counts[col] = count;
    }
}


static PyObject *pack_csc_columns(PyObject *self, PyObject *args) {
    PyObject *o[6]; Py_buffer b[6] = {{0}};
    int height;
    if (!PyArg_ParseTuple(args, "iOOOOOO", &height,
                          &o[0], &o[1], &o[2], &o[3], &o[4], &o[5])) return NULL;
    GET(0, 8, 0); GET(1, 4, 0); GET(2, 4, 0);
    GET(3, 4, 1); GET(4, 8, 1); GET(5, 8, 1);
    Py_ssize_t width = b[3].len / 4, nnz = b[0].len / 8;
    REQUIRE(height >= 0 && width <= INT32_MAX && nnz <= INT32_MAX);
    int words = height / 64 + (height % 64 != 0);
    REQUIRE(shape(&b[1], nnz, 1, 4) && shape(&b[2], width + 1, 1, 4));
    REQUIRE(shape(&b[4], width, 3, 8) && shape(&b[5], width, 2 * (Py_ssize_t)words, 8));
    REQUIRE(csc_offsets(b[2].buf, width, nnz));
    const int32_t *rows = b[1].buf, *ptr = b[2].buf;
    int valid = 1, count = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t col = 0; col < width && valid; ++col) {
        int32_t prev = -1;
        for (int32_t i = ptr[col]; i < ptr[col + 1]; ++i) {
            if (rows[i] <= prev || rows[i] >= height) { valid = 0; break; }
            prev = rows[i];
        }
    }
    if (valid) count = packed_pack((int32_t)width, words, b[0].buf, rows, ptr,
                                  b[3].buf, b[4].buf, b[5].buf);
    Py_END_ALLOW_THREADS
    REQUIRE(valid);
    release(b, 6);
    return PyLong_FromLong(count);
fail:
    release(b, 6); return NULL;
}

static PyObject *pack_mask_columns(PyObject *self, PyObject *args) {
    PyObject *o[7]; Py_buffer b[7] = {{0}};
    if (!PyArg_ParseTuple(args, "OOOOOOO", &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6]))
        return NULL;
    GET(0, 1, 0); GET(1, 1, 0); GET(2, 8, 0); GET(3, 8, 0);
    GET(4, 8, 0); GET(5, 8, 1); GET(6, 4, 1);
    Py_ssize_t width = b[2].len / 8, rows = b[3].len / 8;
    REQUIRE(width <= INT32_MAX && rows <= INT32_MAX);
    Py_ssize_t words = (rows + 63) / 64;
    REQUIRE(shape(&b[4], width, 3, 8) && shape(&b[5], width, 2 * words, 8));
    REQUIRE(shape(&b[6], width, 1, 4));
    if (!width) { release(b, 7); Py_RETURN_NONE; }
    REQUIRE(b[0].len % width == 0);
    Py_ssize_t stride = b[0].len / width;
    REQUIRE(stride > 0 && stride <= INT32_MAX && b[1].len % stride == 0);
    const int64_t *sites = b[2].buf, *sources = b[3].buf;
    for (Py_ssize_t i = 0; i < width; ++i)
        REQUIRE(sites[i] >= 0 && sites[i] < b[1].len / stride);
    for (Py_ssize_t i = 0; i < rows; ++i)
        REQUIRE(sources[i] == -1 || (sources[i] >= 0 && sources[i] / 8 < stride));
    Py_BEGIN_ALLOW_THREADS
    memset(b[5].buf, 0, (size_t)b[5].len);
    packed_pack_binary((int32_t)width, (int32_t)rows, (int32_t)stride,
                       b[0].buf, b[1].buf, sites, sources, b[4].buf, b[5].buf, b[6].buf);
    Py_END_ALLOW_THREADS
    release(b, 7); Py_RETURN_NONE;
fail:
    release(b, 7); return NULL;
}

static int valid_mapping(int32_t index, Py_ssize_t packed, Py_ssize_t fallback) {
    return index >= 0 ? index < packed : -1 - index < fallback;
}

static PyObject *scan_packed_columns(PyObject *self, PyObject *args) {
    PyObject *o[8]; Py_buffer b[8] = {{0}};
    int fused;
    if (!PyArg_ParseTuple(args, "pOOOOOOOO", &fused,
                          &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7])) return NULL;
    GET(0, 4, 0); GET(1, 8, 0); GET(2, 8, 0); GET(3, 8, 0);
    GET(4, 4, 0); GET(5, 4, 0); GET(6, 8, 0); GET(7, 8, 1);
    Py_ssize_t width = b[0].len / 4, packed = b[1].len / 24, height = b[6].len / 8;
    Py_ssize_t nnz = b[3].len / 8, words = (height + 63) / 64, fallback = b[5].len / 4 - 1;
    REQUIRE(width <= INT32_MAX && height <= INT32_MAX && fallback >= 0);
    REQUIRE(shape(&b[2], packed, 2 * words, 8) && shape(&b[7], width, 1, 8));
    REQUIRE(shape(&b[4], nnz, 1, 4) && csc_offsets(b[5].buf, fallback, nnz));
    const int32_t *mapping = b[0].buf, *rows = b[4].buf;
    const uint64_t *planes = b[2].buf;
    int valid = 1;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < width && valid; ++i) valid = valid_mapping(mapping[i], packed, fallback);
    for (Py_ssize_t i = 0; i < nnz && valid; ++i) valid = rows[i] >= 0 && rows[i] < height;
    if (height % 64) {
        for (Py_ssize_t i = 0; i < 2 * packed && valid; ++i)
            valid = (planes[(i + 1) * words - 1] >> (height % 64)) == 0;
    }
    if (valid) scan_impl(fused, (int32_t)width, (int32_t)words, mapping, b[1].buf, planes,
                         b[3].buf, rows, b[5].buf, b[6].buf, b[7].buf);
    Py_END_ALLOW_THREADS
    REQUIRE(valid);
    release(b, 8); Py_RETURN_NONE;
fail:
    release(b, 8); return NULL;
}

static PyObject *gather_packed_columns(PyObject *self, PyObject *args) {
    PyObject *o[10]; Py_buffer b[10] = {{0}};
    int height;
    if (!PyArg_ParseTuple(args, "iOOOOOOOOOO", &height,
                          &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7], &o[8], &o[9]))
        return NULL;
    GET(0, 4, 0); GET(1, 8, 0); GET(2, 8, 0); GET(3, 8, 0); GET(4, 4, 0);
    GET(5, 4, 0); GET(6, 8, 0); GET(7, 4, 0); GET(8, 8, 1); GET(9, 4, 1);
    REQUIRE(height >= 0);
    int words = height / 64 + (height % 64 != 0);
    Py_ssize_t width = b[0].len / 4, packed = b[1].len / 24, fallback = b[5].len / 4 - 1;
    Py_ssize_t nnz = b[3].len / 8, count = b[6].len / 8, out_nnz = b[8].len / 8;
    REQUIRE(fallback >= 0 && shape(&b[2], packed, 2 * (Py_ssize_t)words, 8));
    REQUIRE(shape(&b[4], nnz, 1, 4) && csc_offsets(b[5].buf, fallback, nnz));
    REQUIRE(shape(&b[7], count + 1, 1, 4) && shape(&b[9], out_nnz, 1, 4));
    REQUIRE(csc_offsets(b[7].buf, count, out_nnz));
    const int32_t *mapping = b[0].buf, *ptr = b[5].buf, *out_ptr = b[7].buf;
    const int64_t *columns = b[6].buf;
    const double *values = b[1].buf, *data = b[3].buf;
    const int32_t *rows = b[4].buf;
    const uint64_t *planes = b[2].buf;
    double *out = b[8].buf;
    int32_t *indices = b[9].buf;
    int valid = 1;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t col = 0; col < count; ++col) {
        if (columns[col] < 0 || columns[col] >= width) { valid = 0; break; }
        int32_t index = mapping[columns[col]], dest = out_ptr[col];
        int32_t length = out_ptr[col + 1] - dest;
        if (!valid_mapping(index, packed, fallback)) { valid = 0; break; }
        if (index >= 0) {
            const uint64_t *low = planes + (int64_t)index * 2 * words;
            const uint64_t *high = low + words;
            uint64_t entries = 0;
            for (int word = 0; word < words; ++word) entries += pop64(low[word] | high[word]);
            if (entries != (uint64_t)length ||
                (height % 64 && ((low[words - 1] | high[words - 1]) >> (height % 64)))) {
                valid = 0;
                break;
            }
            if (length) packed_take(words, values + 3 * (int64_t)index, low, out + dest, indices + dest);
        } else {
            int32_t start = ptr[-1 - index];
            if (length != ptr[-index] - start) { valid = 0; break; }
            if (length) {
                memcpy(out + dest, data + start, (size_t)length * sizeof(*out));
                memcpy(indices + dest, rows + start, (size_t)length * sizeof(*indices));
            }
        }
    }
    Py_END_ALLOW_THREADS
    REQUIRE(valid);
    release(b, 10); Py_RETURN_NONE;
fail:
    release(b, 10); return NULL;
}
