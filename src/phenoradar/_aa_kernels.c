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

static PyMethodDef methods[] = {{"decode", decode, METH_VARARGS, NULL},
    {"center", center, METH_VARARGS, NULL}, {"csc", csc, METH_VARARGS, NULL},
    {"packed_csc", packed_csc, METH_VARARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_aa_kernels", NULL, -1, methods};
PyMODINIT_FUNC PyInit__aa_kernels(void) { return PyModule_Create(&module); }
