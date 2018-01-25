#pragma once

/* boost includes */
#include <boost/math/quaternion.hpp>

/* opencv includes */
#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

/* pcl headers */
#include <pcl/point_types.h>

/* sys headers */
#include <math.h>
#include <cassert>
#include <iostream>

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

    /* FLOAT_EPSILON */
    const float epsilon = 1.192092896e-07f;

public:
    /* constructor for dual quaternion from rotation unit quaternion and translation quaternion */
    DualQuaternion(boost::math::quaternion<T> rotation, boost::math::quaternion<T> translation)
        : real(rotation), dual(translation) {}

    /* constructor for dual quaternion from rotation quaternion and translation vector */
    DualQuaternion(boost::math::quaternion<T> rotation, cv::Vec3f translation) {
        real = normalize(rotation);
        dual = (boost::math::quaternion<T>(0, translation[0], translation[1], translation[2]) * real) * 0.5f;
    }

    /* constructor for dual quaternion from Euler angles (yaw, pitch, roll) and translation vector */
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

    /* constructor for dual quaternion from Euler-Rodrigues vector and translation vector */
    DualQuaternion(cv::Vec3f rodrigues, cv::Vec3f translation) {
        auto theta          = 2 * atan(cv::norm(rodrigues));  // rotation angle
        auto axis           = rodrigues / theta;
        auto axisNormalised = axis / cv::norm(axis);  // normalised rotation axis

        auto s  = sin(0.5 * theta);
        auto q1 = s * axisNormalised(0);
        auto q2 = s * axisNormalised(1);
        auto q3 = s * axisNormalised(2);
        auto q4 = cos(0.5 * theta);

        boost::math::quaternion<float> rotation(q4, q1, q2, q3);  // rotation quaternion
        DualQuaternion<T> dq(normalize(rotation), translation);

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

    /* TODO: allow 0.5*dq to be used */
    DualQuaternion<T> operator*(T scale) { return DualQuaternion<T>(real, dual * scale); }

    DualQuaternion<T>& operator*=(T scale) {
        dual *= scale;
        return *this;
    }

    DualQuaternion<T> operator*(const DualQuaternion<T>& other) {
        return DualQuaternion<T>(real * other.getReal(), real * other.getDual() + dual * other.getReal());
    }

    DualQuaternion<T>& operator*=(const DualQuaternion<T>& other) {
        /* Make sure the real is updated after dual to avoid using updated real */
        dual = real * other.getDual() + dual * other.getReal();
        real *= other.getReal();
    }

    DualQuaternion<T> conj() { return DualQuaternion<T>(boost::math::conj(real), boost::math::conj(dual)); }

    DualQuaternion<T>& normalize() {
        T magnitude = sqrtf(dotProduct(real, real));
        assert(magnitude > epsilon);
        real *= (1.0f / magnitude);
        return *this;
    }

    ~DualQuaternion() {}

    T getRoll() const {
        boost::math::quaternion<T> q = real;

        // roll (x-axis rotation)
        float sinr = +2.0 * (q.R_component_1() * q.R_component_2() + q.R_component_3() * q.R_component_4());
        float cosr = +1.0 - 2.0 * (q.R_component_2() * q.R_component_2() + q.R_component_3() * q.R_component_3());
        T roll     = atan2(sinr, cosr);

        if (roll > M_PI) {
            roll -= M_PI_2;
        }

        return roll;
    }

    T getPitch() const {
        boost::math::quaternion<T> q = real;

        // pitch (y-axis rotation)
        T pitch;
        float sinp = +2.0 * (q.R_component_1() * q.R_component_3() - q.R_component_4() * q.R_component_2());

        if (fabs(sinp) >= 1)
            pitch = copysign(M_PI / 2, sinp);  // use 90 degrees if out of range
        else {
            pitch = asin(sinp);
        }

        return pitch;
    }

    T getYaw() const {
        boost::math::quaternion<T> q = real;

        // yaw (z-axis rotation)
        float siny = +2.0 * (q.R_component_1() * q.R_component_4() + q.R_component_2() * q.R_component_3());
        float cosy = +1.0 - 2.0 * (q.R_component_3() * q.R_component_3() + q.R_component_4() * q.R_component_4());
        T yaw      = atan2(siny, cosy);

        if (yaw > M_PI) {
            yaw -= M_PI_2;
        }

        return yaw;
    }

    cv::Vec3f getEulerAngles() const { return cv::Vec3f(getRoll(), getPitch(), getYaw()); }

    cv::Vec3f getRodrigues() {
        auto q     = cv::Vec3f(real.R_component_2(), real.R_component_3(), real.R_component_4());
        auto norm  = cv::norm(q);
        auto theta = 2 * acos(real.R_component_1());

        return tan(0.5 * theta) * q / norm;
    }

    pcl::PointXYZ transformVertex(pcl::PointXYZ v) {
        cv::Vec3f vect(v.x, v.y, v.z);

        cv::Vec3f realVect = cv::Vec3f(real.R_component_2(), real.R_component_3(), real.R_component_4());
        cv::Vec3f dualVect = cv::Vec3f(dual.R_component_2(), dual.R_component_3(), dual.R_component_4());

        cv::Vec3f result =
            vect + 2.f * realVect.cross(realVect.cross(vect) + real.R_component_1() * vect) +
            2.f * (real.R_component_1() * dualVect - dual.R_component_1() * realVect + realVect.cross(dualVect));

        return pcl::PointXYZ(result[0], result[1], result[2]);
    }

    pcl::Normal transformNormal(pcl::Normal n) {
        cv::Vec3f vect(n.data_c[0], n.data_c[1], n.data_c[2]);

        cv::Vec3f realVect = cv::Vec3f(real.R_component_2(), real.R_component_3(), real.R_component_4());
        cv::Vec3f dualVect = cv::Vec3f(dual.R_component_2(), dual.R_component_3(), dual.R_component_4());

        cv::Vec3f result =
            vect + 2.f * realVect.cross(realVect.cross(vect) + real.R_component_1() * vect) +
            2.f * (real.R_component_1() * dualVect - dual.R_component_1() * realVect + realVect.cross(dualVect));

        return pcl::Normal(result[0], result[1], result[2]);
    }

    friend std::ostream& operator<<(std::ostream& os, const DualQuaternion<T>& dq) {
        return os << "real: " << dq.getReal() << "\ndual: " << dq.getDual() << std::endl;
    }
};
