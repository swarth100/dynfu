#pragma once
#include <boost/math/quaternion.hpp>
#include <iostream>

#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

#include <math.h>
#include <cassert>

template <class T>
class DualQuaternion {
private:
    /* Rotational part */
    boost::math::quaternion<T> real;
    /* Translation (displacement) part */
    boost::math::quaternion<T> dual;

    T dotProduct(boost::math::quaternion<T> q1, boost::math::quaternion<T> q2) {
        return q1.R_component_1() * q2.R_component_1() + q1.R_component_2() * q2.R_component_2() +
               q1.R_component_3() * q2.R_component_3() + q1.R_component_4() * q2.R_component_4();
    }

    boost::math::quaternion<T> normalize(boost::math::quaternion<T> q) { return q / boost::math::norm(q); }

    /*FLOAT_EPSILON*/
    const float epsilon = 1.192092896e-07f;

public:
    DualQuaternion(boost::math::quaternion<T> rotation, boost::math::quaternion<T> translation)
        : real(rotation), dual(translation) {}

    DualQuaternion(boost::math::quaternion<T> rotation, cv::Vec3f translation) {
        real = normalize(rotation);
        dual = (boost::math::quaternion<T>(0, translation[0], translation[1], translation[2]) * real) * 0.5f;
    }

    DualQuaternion(T yaw, T pitch, T roll, T x, T y, T z) {
        T cy = cos(yaw * 0.5);
        T sy = sin(yaw * 0.5);
        T cr = cos(roll * 0.5);
        T sr = sin(roll * 0.5);
        T cp = cos(pitch * 0.5);
        T sp = sin(pitch * 0.5);

        T qw = cy * cr * cp + sy * sr * sp;
        T qx = cy * sr * cp - sy * cr * sp;
        T qy = cy * cr * sp + sy * sr * cp;
        T qz = sy * cr * cp - cy * sr * sp;
        boost::math::quaternion<T> rotation(qw, qx, qy, qz);
        DualQuaternion<T> dq(rotation, cv::Vec3f(x, y, z));
        real = dq.getReal();
        dual = dq.getDual();
    }

    boost::math::quaternion<T> getReal() const { return real; }

    boost::math::quaternion<T> getDual() const { return dual; }

    boost::math::quaternion<T> getRotation() const { return real; }

    cv::Vec3f getTranslation() const {
        boost::math::quaternion<T> q = (dual * 2.0f) * boost::math::conj(real);
        return cv::Vec3f(q.R_component_2(), q.R_component_3(), q.R_component_4());
    }

    DualQuaternion<T> operator+(const DualQuaternion<T>& other) {
        return DualQuaternion<T>(real + other.getReal(), dual + other.getDual());
    }

    DualQuaternion<T>& operator+=(const DualQuaternion<T>& other) {
        real += other.getReal();
        dual += other.getDual();
        return *this;
    }

    DualQuaternion<T> operator-(const DualQuaternion<T>& other) {
        return DualQuaternion<T>(real - other.getReal(), dual - other.getDual());
    }

    DualQuaternion<T>& operator-=(const DualQuaternion<T>& other) {
        real -= other.getReal();
        dual -= other.getDual();
        return *this;
    }

    /* TODO allow 0,5 * dq to be used */

    DualQuaternion<T> operator*(T scale) { return DualQuaternion<T>(real * scale, real * scale); }

    DualQuaternion<T>& operator*=(T scale) {
        real *= scale;
        dual *= scale;
        return *this;
    }

    /* TODO confirm this
     * http://wscg.zcu.cz/wscg2012/short/A29-full.pdf */

    DualQuaternion<T> operator*(const DualQuaternion<T>& other) {
        return DualQuaternion<T>(real * other.getReal(), real * other.getDual() + dual * other.getReal());
    }

    DualQuaternion<T>& operator*=(const DualQuaternion<T>& other) {
        /* Make sure the real is updated after dual to avoid using updated real */
        dual = real * other.getDual() + dual * other.getReal();
        real *= other.getReal();
    }

    DualQuaternion<T>& normalize() {
        T magnitude = dotProduct(real, dual);
        assert(magnitude > epsilon);
        real *= (1.0f / magnitude);
        return *this;
    }
    ~DualQuaternion() {}

    friend std::ostream& operator<<(std::ostream& os, const DualQuaternion<T>& dq) {
        return os << "R: " << dq.getReal() << " D:" << dq.getDual() << std::endl;
    }
};
