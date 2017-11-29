#ifndef DYNFU_NODE_HPP
#define DYNFU_NODE_HPP

/* dynfu includes */
#include <dynfu/utils/dual_quaternion.hpp>

/* opencv includes */
#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

/* sys headers */
#include <memory>
#include <vector>

/*
 * node of the warp field
 * the state of the warp field at a given time list defined by the
 * set of n deformation nodes
 *
 * dg_v
 * position of the node in space; will be used when computing k-NN for
 * warping points
 *
 * dg_se3
 * transformation
 *
 * dg_w
 * radial basis weight which controls the extent of transformation
 */
class Node {
public:
    Node(cv::Vec3f position, std::shared_ptr<DualQuaternion<float>> transformation, float weight);
    ~Node();

    cv::Vec3f getPosition();
    float getWeight();

    double* getParams();

    std::shared_ptr<DualQuaternion<float>>& getTransformation();
    void setTraslation(cv::Vec3f translation);
    void setWeight();
    void setTransformation(std::shared_ptr<DualQuaternion<float>> transformation);
    void setTranslation(cv::Vec3f translation);
    void setRotation(boost::math::quaternion<float> real);

private:
    cv::Vec3f dg_v;
    std::shared_ptr<DualQuaternion<float>> dg_se3;
    float dg_w;

    double* params;
};

/* DYNFU_NODE_HPP */
#endif
