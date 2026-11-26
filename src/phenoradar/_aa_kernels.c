/* Exact buffer kernels. No NumPy ABI and no floating-point reassociation. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

static int get(PyObject *o, Py_buffer *b, int size, int writable) {
    if (PyObject_GetBuffer(o, b, PyBUF_C_CONTIGUOUS | (writable ? PyBUF_WRITABLE : 0)) < 0)
        return 0;
    if (b->itemsize != size) {
        PyErr_SetString(PyExc_ValueError, "Unexpected kernel buffer dtype");
        return 0;
    }
    return 1;
}
#define GET(i, size, write) if (!get(o[i], &b[i], size, write)) goto fail
#define REQUIRE(expr) if (!(expr)) { PyErr_SetString(PyExc_ValueError, "Invalid kernel buffer shape/index"); goto fail; }
static void release(Py_buffer *b, int n) {
    for (int i = 0; i < n; ++i) if (b[i].obj) PyBuffer_Release(&b[i]);
}
static int shape(Py_buffer *b, Py_ssize_t rows, Py_ssize_t cols, int size) {
    if (rows < 0 || cols < 0 || (cols && rows > PY_SSIZE_T_MAX / size / cols)) return 0;
    return b->len == rows * cols * size;
}
static unsigned pop(unsigned x) {
    x = x - ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    return (x + (x >> 4)) & 15;
}

static unsigned pop64(uint64_t x) {
    x -= (x >> 1) & UINT64_C(0x5555555555555555);
    x = (x & UINT64_C(0x3333333333333333)) + ((x >> 2) & UINT64_C(0x3333333333333333));
    x = (x + (x >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return (unsigned)((x * UINT64_C(0x0101010101010101)) >> 56);
}

static PyObject *masked_counts(PyObject *self, PyObject *args) {
    PyObject *o[3]; Py_buffer b[3] = {{0}};
    if (!PyArg_ParseTuple(args, "OOO", &o[0], &o[1], &o[2])) return NULL;
    GET(0, 1, 0); GET(1, 1, 0); GET(2, 4, 1);
    Py_ssize_t width = b[1].len, rows = b[2].len / 4;
    REQUIRE(width <= INT32_MAX / 8 && shape(&b[0], rows, width, 1));
    const uint8_t *packed = b[0].buf, *mask = b[1].buf;
    int32_t *out = b[2].buf;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t row = 0; row < rows; ++row) {
        const uint8_t *bits = packed + row * width;
        int32_t count = 0;
        Py_ssize_t col = 0;
        for (; col + 8 <= width; col += 8) {
            uint64_t a, m;
            memcpy(&a, bits + col, 8);
            memcpy(&m, mask + col, 8);
            count += (int32_t)pop64(a & m);
        }
        for (; col < width; ++col) count += (int32_t)pop(bits[col] & mask[col]);
        out[row] = count;
    }
    Py_END_ALLOW_THREADS
    release(b, 3); Py_RETURN_NONE;
fail:
    release(b, 3); return NULL;
}

static PyObject *decode(PyObject *self, PyObject *args) {
    PyObject *o[4]; Py_buffer b[4] = {{0}};
    Py_ssize_t nr, width; uint64_t *starts = NULL, *prefix = NULL;
    if (!PyArg_ParseTuple(args, "OnnOOO", &o[0], &nr, &width, &o[1], &o[2], &o[3])) return NULL;
    GET(0, 1, 0); GET(1, 8, 0); GET(2, 8, 0); GET(3, 4, 1);
    REQUIRE(nr >= 0 && width >= 0 && width < PY_SSIZE_T_MAX - 7);
    Py_ssize_t nb = (width + 7) / 8;
    REQUIRE(nb == 0 || nr <= PY_SSIZE_T_MAX / 3 / nb);
    Py_ssize_t end = nr * 3 * nb, nrows = b[1].len / 8, ncols = b[2].len / 8;
    REQUIRE(end <= b[0].len && (nrows == 0 || ncols <= PY_SSIZE_T_MAX / nrows / 4));
    REQUIRE(b[3].len == nrows * ncols * 4);
    uint8_t *data = b[0].buf; int64_t *rows = b[1].buf, *cols = b[2].buf;
    for (Py_ssize_t r = 0; r < nrows; ++r) REQUIRE(rows[r] >= 0 && rows[r] < nr);
    for (Py_ssize_t c = 0; c < ncols; ++c) REQUIRE(cols[c] >= 0 && cols[c] < width);
    starts = PyMem_Calloc((size_t)nr + 1, sizeof(uint64_t));
    prefix = PyMem_Calloc((size_t)nb + 1, sizeof(uint64_t));
    if (!starts || !prefix) { PyErr_NoMemory(); goto fail; }
    uint32_t *out = b[3].buf; int invalid = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t r = 0; r < nr; ++r) {
        uint8_t *exceptions = data + (r * 3 + 2) * nb;
        starts[r + 1] = starts[r];
        for (Py_ssize_t j = 0; j < nb; ++j) starts[r + 1] += pop(exceptions[j]);
    }
    for (Py_ssize_t r = 0; r < nrows; ++r) {
        uint8_t *planes = data + rows[r] * 3 * nb;
        prefix[0] = 0;
        for (Py_ssize_t j = 0; j < nb; ++j) prefix[j + 1] = prefix[j] + pop(planes[2 * nb + j]);
        for (Py_ssize_t c = 0; c < ncols; ++c) {
            Py_ssize_t j = cols[c] / 8; unsigned bit = 1U << (cols[c] % 8);
            uint32_t value = planes[j] & bit ? 0x3f800000 : 0;
            if (planes[nb + j] & bit) value = 0x7fc00000;
            if (planes[2 * nb + j] & bit) {
                uint64_t index = starts[rows[r]] + prefix[j] + pop(planes[2 * nb + j] & (bit - 1));
                if (index >= (uint64_t)((b[0].len - end) / 4)) invalid = 1;
                else memcpy(&value, data + end + index * 4, 4);
            }
            out[r * ncols + c] = value;
        }
    }
    Py_END_ALLOW_THREADS
    if (invalid) { PyErr_SetString(PyExc_ValueError, "Truncated packed frequencies"); goto fail; }
    PyMem_Free(starts); PyMem_Free(prefix); release(b, 4); Py_RETURN_NONE;
fail:
    PyMem_Free(starts); PyMem_Free(prefix); release(b, 4); return NULL;
}

#include "_aa_decode_into.h"

static PyObject *center(PyObject *self, PyObject *args) {
    PyObject *o[6]; Py_buffer b[6] = {{0}}; Py_ssize_t n, raw_width, mass_width;
    if (!PyArg_ParseTuple(args, "OOOOOOnnn", &o[0], &o[1], &o[2], &o[3], &o[4], &o[5],
                          &n, &raw_width, &mass_width)) return NULL;
    GET(0, 4, 0); GET(1, 8, 0); GET(2, 8, 0); GET(3, 8, 0); GET(4, 8, 0); GET(5, 8, 1);
    Py_ssize_t k = b[1].len / 8;
    REQUIRE(n >= 0 && raw_width >= 0 && mass_width >= 0);
    REQUIRE(shape(&b[0], n, raw_width, 4) && b[2].len == k * 8 && b[3].len == k * 8);
    REQUIRE(shape(&b[4], n, mass_width, 8) && shape(&b[5], n, k, 8));
    float *raw = b[0].buf; int64_t *cols = b[1].buf, *sites = b[2].buf;
    double *mu = b[3].buf, *mass = b[4].buf, *out = b[5].buf;
    for (Py_ssize_t j = 0; j < k; ++j) REQUIRE(cols[j] >= 0 && cols[j] < raw_width && sites[j] >= 0 && sites[j] < mass_width);
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) for (Py_ssize_t j = 0; j < k; ++j) {
        float q = raw[i * raw_width + cols[j]];
        if (isnan(q)) q = 0;
        else if (isinf(q)) q = copysignf(FLT_MAX, q);
        double product = mass[i * mass_width + sites[j]] * mu[j];
        out[i * k + j] = (double)q - product;
    }
    Py_END_ALLOW_THREADS
    release(b, 6); Py_RETURN_NONE;
fail:
    release(b, 6); return NULL;
}

static PyObject *csc(PyObject *self, PyObject *args) {
    PyObject *o[6]; Py_buffer b[6] = {{0}}; Py_ssize_t n, k;
    if (!PyArg_ParseTuple(args, "OOOOOOnn", &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &n, &k)) return NULL;
    GET(0, 8, 0); GET(1, 8, 0); GET(2, 8, 0); GET(3, 8, 1); GET(4, 4, 1); GET(5, 8, 1);
    Py_ssize_t m = b[2].len / 8;
    REQUIRE(shape(&b[0], n, k, 8) && b[1].len / 8 == k);
    REQUIRE(shape(&b[3], m, k, 8) && shape(&b[4], m, k, 4) && b[5].len / 8 == k);
    double *values = b[0].buf, *shift = b[1].buf, *data = b[3].buf;
    int64_t *rows = b[2].buf, *ends = b[5].buf; int32_t *indices = b[4].buf;
    for (Py_ssize_t i = 0; i < m; ++i) REQUIRE(rows[i] >= 0 && rows[i] < n);
    REQUIRE(m <= INT32_MAX && k <= INT32_MAX && (m == 0 || k <= INT32_MAX / m));
    Py_ssize_t count = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t j = 0; j < k; ++j) {
        for (Py_ssize_t i = 0; i < m; ++i) {
            double value = values[rows[i] * k + j] + shift[j];
            if (value != 0) { data[count] = value; indices[count] = (int32_t)i; ++count; }
        }
        ends[j] = count;
    }
    Py_END_ALLOW_THREADS
    release(b, 6); return PyLong_FromSsize_t(count);
fail:
    release(b, 6); return NULL;
}
/* At most 20 nonnegative float32 values spanning <=24 binades need at most
   24 + 24 + ceil(log2(20)) = 53 significant bits. Their float64 sum is exact
   in every order. Sums safely inside the existing unit-mass clipping guard
   also produce exactly 1. With <=2 nonzero terms, every reduction order is
   equivalent. Other sites retain the original NumPy reduction. */
static PyObject *centering_mass(PyObject *self, PyObject *args) {
    PyObject *o[5]; Py_buffer b[5] = {{0}};
    Py_ssize_t n, raw_width, nsites;
    if (!PyArg_ParseTuple(args, "OOOOOnnn", &o[0], &o[1], &o[2], &o[3], &o[4],
                          &n, &raw_width, &nsites)) return NULL;
    GET(0, 4, 0); GET(1, 8, 0); GET(2, 8, 0); GET(3, 8, 1); GET(4, 1, 1);
    Py_ssize_t k = b[1].len / 8;
    REQUIRE(n >= 0 && raw_width >= 0 && nsites >= 0);
    REQUIRE(shape(&b[0], n, raw_width, 4) && shape(&b[3], n, nsites, 8));
    REQUIRE(b[2].len == k * 8 && b[4].len == nsites);
    const float *raw = b[0].buf;
    const int64_t *columns = b[1].buf, *sites = b[2].buf;
    double *mass = b[3].buf;
    uint8_t *fallback = b[4].buf;
    Py_ssize_t count = 0;
    for (Py_ssize_t j = 0; j < k; ++j) {
        REQUIRE(columns[j] >= 0 && columns[j] < raw_width && sites[j] >= 0 && sites[j] < nsites);
        REQUIRE(j == 0 || (columns[j] > columns[j - 1] && sites[j] >= sites[j - 1]));
        count = j && sites[j] == sites[j - 1] ? count + 1 : 1;
        REQUIRE(count <= 20);
    }
    Py_ssize_t failures = 0, left = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t site = 0; site < nsites; ++site) {
        Py_ssize_t right = left;
        while (right < k && sites[right] == site) ++right;
        fallback[site] = 0;
        for (Py_ssize_t i = 0; i < n; ++i) {
            double sum = 0, minimum = DBL_MAX, maximum = 0;
            int valid = 1, terms = 0;
            for (Py_ssize_t j = left; j < right; ++j) {
                float q = raw[i * raw_width + columns[j]];
                if (isnan(q)) continue;
                if (q == 0 && !signbit(q)) continue;
                if (!isfinite(q) || q < 0 || signbit(q)) { valid = 0; break; }
                double value = q;
                ++terms;
                sum += value;
                if (value < minimum) minimum = value;
                if (value > maximum) maximum = value;
            }
            int clipped = fabs(sum - 1.0) < 1e-7 - 64 * DBL_EPSILON;
            if (!valid || (!clipped && terms > 2 && minimum < maximum / 16777216.0)) {
                fallback[site] = 1; ++failures; break;
            }
            mass[i * nsites + site] = fabs(sum - 1.0) <= 1e-7 ? 1 : sum;
        }
        left = right;
    }
    Py_END_ALLOW_THREADS
    release(b, 5); return PyLong_FromSsize_t(failures);
fail:
    release(b, 5); return NULL;
}

static PyObject *packed_csc(PyObject *self, PyObject *args) {
    PyObject *o[8]; Py_buffer b[8] = {{0}}; Py_ssize_t width, mass_rows;
    if (!PyArg_ParseTuple(args, "OOOOOOOOnn", &o[0], &o[1], &o[2], &o[3],
                          &o[4], &o[5], &o[6], &o[7], &width, &mass_rows)) return NULL;
    GET(0, 1, 0); GET(1, 1, 0); GET(2, 8, 0); GET(3, 8, 0);
    GET(4, 8, 0); GET(5, 8, 1); GET(6, 4, 1); GET(7, 4, 1);
    REQUIRE(width >= 0 && width <= PY_SSIZE_T_MAX - 7);
    Py_ssize_t k = b[2].len / 8, m = b[3].len / 8, bytes = (width + 7) / 8;
    REQUIRE(shape(&b[0], k, bytes, 1) && shape(&b[1], mass_rows, bytes, 1));
    REQUIRE(shape(&b[4], k, 3, 8) && shape(&b[5], m, k, 8) && shape(&b[6], m, k, 4));
    REQUIRE(b[7].len / 4 == k + 1 && m <= INT32_MAX && k <= INT32_MAX);
    REQUIRE(m == 0 || k <= INT32_MAX / m);
    const uint8_t *ones = b[0].buf, *mass = b[1].buf;
    const int64_t *sites = b[2].buf, *rows = b[3].buf;
    const double *categories = b[4].buf;
    double *data = b[5].buf;
    int32_t *indices = b[6].buf, *indptr = b[7].buf;
    for (Py_ssize_t j = 0; j < k; ++j) REQUIRE(sites[j] >= 0 && sites[j] < mass_rows);
    for (Py_ssize_t i = 0; i < m; ++i) REQUIRE(rows[i] >= -1 && rows[i] < width);
    int32_t count = 0;
    indptr[0] = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t j = 0; j < k; ++j) {
        const uint8_t *state = ones + j * bytes, *known = mass + sites[j] * bytes;
        for (Py_ssize_t i = 0; i < m; ++i) {
            int64_t row = rows[i];
            int category = 1;
            if (row >= 0) {
                unsigned bit = 1U << (row % 8);
                if (state[row / 8] & bit) category = 2;
                else if (known[row / 8] & bit) category = 0;
            }
            double value = categories[j * 3 + category];
            if (value != 0) { data[count] = value; indices[count] = (int32_t)i; ++count; }
        }
        indptr[j + 1] = count;
    }
    Py_END_ALLOW_THREADS
    release(b, 8); return PyLong_FromLong(count);
fail:
    release(b, 8); return NULL;
}

static int csc_offsets(const int32_t *indptr, Py_ssize_t columns, Py_ssize_t nnz) {
    if (indptr[0] != 0 || indptr[columns] != nnz) return 0;
    for (Py_ssize_t col = 0; col < columns; ++col) {
        if (indptr[col] > indptr[col + 1]) return 0;
    }
    return 1;
}

static PyObject *interleave_csc(PyObject *self, PyObject *args) {
    PyObject *o[10]; Py_buffer b[10] = {{0}};
    if (!PyArg_UnpackTuple(args, "interleave_csc", 10, 10,
                           &o[0], &o[1], &o[2], &o[3], &o[4],
                           &o[5], &o[6], &o[7], &o[8], &o[9])) return NULL;
    GET(0, 8, 0); GET(1, 4, 0); GET(2, 4, 0);
    GET(3, 8, 0); GET(4, 4, 0); GET(5, 4, 0);
    GET(6, 8, 0); GET(7, 8, 1); GET(8, 4, 1); GET(9, 4, 1);
    REQUIRE(b[2].len >= 4 && b[5].len >= 4);
    Py_ssize_t left = b[2].len / 4 - 1, right = b[5].len / 4 - 1;
    Py_ssize_t left_nnz = b[0].len / 8, right_nnz = b[3].len / 8;
    Py_ssize_t width = b[6].len / 8, nnz = b[7].len / 8;
    REQUIRE(left + right == width && left_nnz + right_nnz == nnz && nnz <= INT32_MAX);
    REQUIRE(b[1].len / 4 == left_nnz && b[4].len / 4 == right_nnz);
    REQUIRE(b[8].len / 4 == nnz && b[9].len / 4 == width + 1);

    const int32_t *pointers[] = {b[2].buf, b[5].buf};
    REQUIRE(csc_offsets(pointers[0], left, left_nnz) && csc_offsets(pointers[1], right, right_nnz));
    const double *data[] = {b[0].buf, b[3].buf};
    const int32_t *indices[] = {b[1].buf, b[4].buf};
    const int64_t *order = b[6].buf;
    double *out_data = b[7].buf;
    int32_t *out_indices = b[8].buf, *out_indptr = b[9].buf;
    Py_ssize_t offset = 0;
    int valid = 1;
    out_indptr[0] = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t dest = 0; dest < width; ++dest) {
        int64_t col = order[dest];
        if (col < 0 || col >= width) { valid = 0; break; }
        int source = col >= left;
        if (source) col -= left;
        int32_t start = pointers[source][col], length = pointers[source][col + 1] - start;
        if (length > nnz - offset) { valid = 0; break; }
        if (length) {
            memcpy(out_data + offset, data[source] + start, (size_t)length * sizeof(double));
            memcpy(out_indices + offset, indices[source] + start, (size_t)length * sizeof(int32_t));
        }
        offset += length;
        out_indptr[dest + 1] = (int32_t)offset;
    }
    Py_END_ALLOW_THREADS
    REQUIRE(valid && offset == nnz);
    release(b, 10); Py_RETURN_NONE;
fail:
    release(b, 10); return NULL;
}

#include "_aa_packed.h"
#include "_aa_statistics.h"

static PyMethodDef methods[] = {{"decode", decode, METH_VARARGS, NULL},
    {"decode_compact", decode_compact, METH_VARARGS, NULL},
    {"center", center, METH_VARARGS, NULL}, {"csc", csc, METH_VARARGS, NULL},
    {"centering_mass", centering_mass, METH_VARARGS, NULL},
    {"packed_csc", packed_csc, METH_VARARGS, NULL},
    {"interleave_csc", interleave_csc, METH_VARARGS, NULL},
    {"pack_csc_columns", pack_csc_columns, METH_VARARGS, NULL},
    {"pack_mask_columns", pack_mask_columns, METH_VARARGS, NULL},
    {"scan_packed_columns", scan_packed_columns, METH_VARARGS, NULL},
    {"prepare_packed_scan", prepare_packed_scan, METH_VARARGS, NULL},
    {"scan_validated_packed_columns", scan_validated_packed_columns, METH_VARARGS, NULL},
    {"gather_packed_columns", gather_packed_columns, METH_VARARGS, NULL},
    {"masked_counts", masked_counts, METH_VARARGS, NULL},
    {"observed_state_statistics", compact_statistics_kernel, METH_VARARGS, NULL},
    {"compact_minor_support", compact_minor_support, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_aa_kernels", NULL, -1, methods};
PyMODINIT_FUNC PyInit__aa_kernels(void) { return PyModule_Create(&module); }
