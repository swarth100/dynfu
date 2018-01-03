#pragma once

#include <cuda_runtime.h>
#include <math.h>

typedef struct {
    float x0, x1, x2, x3, x4, x5, x6, x7;
} float8;

static __inline__ __host__ __device__ float8 make_float8(float x0, float x1, float x2, float x3, float x4, float x5,
                                                         float x6, float x7) {
    float8 t;
    t.x0 = x0;
    t.x1 = x1;
    t.x2 = x2;
    t.x3 = x3;
    t.x4 = x4;
    t.x5 = x5;
    t.x6 = x6;
    t.x7 = x7;
    return t;
}

/////////////// Scalar-wise vector add ////////////////////
__inline__ __host__ __device__ float8 operator+(float8 v0, float8 v1) {
    return make_float8(v0.x0 + v1.x0, v0.x1 + v1.x1, v0.x2 + v1.x2, v0.x3 + v1.x3, v0.x4 + v1.x4, v0.x5 + v1.x5,
                       v0.x6 + v1.x6, v0.x7 + v1.x7);
};

/////////////// Scalar-wise vector subtract ////////////////////
__inline__ __host__ __device__ float8 operator-(float8 v0, float8 v1) {
    return make_float8(v0.x0 - v1.x0, v0.x1 - v1.x1, v0.x2 - v1.x2, v0.x3 - v1.x3, v0.x4 - v1.x4, v0.x5 - v1.x5,
                       v0.x6 - v1.x6, v0.x7 - v1.x7);
}

/////////////// Scalar-wise vector multiply ////////////////////
__inline__ __host__ __device__ float8 operator*(float8 v0, float8 v1) {
    return make_float8(v0.x0 * v1.x0, v0.x1 * v1.x1, v0.x2 * v1.x2, v0.x3 * v1.x3, v0.x4 * v1.x4, v0.x5 * v1.x5,
                       v0.x6 * v1.x6, v0.x7 * v1.x7);
}

/////////////// Scalar-wise vector divide ////////////////////
__inline__ __host__ __device__ float8 operator/(float8 v0, float8 v1) {
    return make_float8(v0.x0 / v1.x0, v0.x1 / v1.x1, v0.x2 / v1.x2, v0.x3 / v1.x3, v0.x4 / v1.x4, v0.x5 / v1.x5,
                       v0.x6 / v1.x6, v0.x7 / v1.x7);
}

/////////////// += ////////////////////
__inline__ __host__ __device__ void operator+=(float8& v0, float8 v1) {
    v0.x0 += v1.x0;
    v0.x1 += v1.x1;
    v0.x2 += v1.x2;
    v0.x3 += v1.x3;
    v0.x4 += v1.x4;
    v0.x5 += v1.x5;
    v0.x6 += v1.x6;
    v0.x7 += v1.x7;
}

__inline__ __host__ __device__ void operator+=(float8& v0, float x) {
    v0.x0 += x;
    v0.x1 += x;
    v0.x2 += x;
    v0.x3 += x;
    v0.x4 += x;
    v0.x5 += x;
    v0.x6 += x;
    v0.x7 += x;
}

/////////////// -= ////////////////////
__inline__ __host__ __device__ void operator-=(float8& v0, float8 v1) {
    v0.x0 -= v1.x0;
    v0.x1 -= v1.x1;
    v0.x2 -= v1.x2;
    v0.x3 -= v1.x3;
    v0.x4 -= v1.x4;
    v0.x5 -= v1.x5;
    v0.x6 -= v1.x6;
    v0.x7 -= v1.x7;
}

__inline__ __host__ __device__ void operator-=(float8& v0, float x) {
    v0.x0 -= x;
    v0.x1 -= x;
    v0.x2 -= x;
    v0.x3 -= x;
    v0.x4 -= x;
    v0.x5 -= x;
    v0.x6 -= x;
    v0.x7 -= x;
}

/////////////// Multiply by a scalar ////////////////////
__inline__ __host__ __device__ float8 operator*(float x, float8 v) {
    return make_float8(v.x0 * x, v.x1 * x, v.x2 * x, v.x3 * x, v.x4 * x, v.x5 * x, v.x6 * x, v.x7 * x);
}

__inline__ __host__ __device__ float8 operator*(float8 v, float x) {
    return make_float8(v.x0 * x, v.x1 * x, v.x2 * x, v.x3 * x, v.x4 * x, v.x5 * x, v.x6 * x, v.x7 * x);
}

/////////////// Divide with a scalar ////////////////////
__inline__ __host__ __device__ float8 operator/(float x, float8 v) {
    return make_float8(v.x0 / x, v.x1 / x, v.x2 / x, v.x3 / x, v.x4 / x, v.x5 / x, v.x6 / x, v.x7 / x);
}

__inline__ __host__ __device__ float8 operator/(float8 v, float x) {
    return make_float8(v.x0 / x, v.x1 / x, v.x2 / x, v.x3 / x, v.x4 / x, v.x5 / x, v.x6 / x, v.x7 / x);
}

__inline__ __host__ __device__ float dot(float8 v0, float8 v1) {
    return v0.x0 * v1.x0 + v0.x1 * v1.x1 + v0.x2 * v1.x2 + v0.x3 * v1.x3 + v0.x4 * v1.x4 + v0.x5 * v1.x5 +
           v0.x6 * v1.x6 + +v0.x7 * v1.x7;
}

__inline__ __host__ __device__ float length(float8 v) { return sqrtf(dot(v, v)); }

__inline__ __host__ __device__ float8 normalize(float8 v) { return v * 1.f / sqrtf(dot(v, v)); }

#if defined(__CUDA_ARCH__)
#define __CONDITIONAL_UNROLL__ #pragma unroll
#else
#define __CONDITIONAL_UNROLL__
#endif
