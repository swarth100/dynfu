#ifndef DYNFU_NODE_HPP
#define DYNFU_NODE_HPP

/* dynfu includes */
#include <dynfu/utils/dual_quaternion.hpp>

/* opencv includes */
#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

/* pcl includes */
#include <pcl/point_types.h>

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
    /* constructor */
    Node(pcl::PointXYZ position, std::shared_ptr<DualQuaternion<float>> transformation, float radialBasisWeight);
    /* destructor */
    ~Node();

    /* get dg_v of a deformation node */
    pcl::PointXYZ getPosition();

    /* get the dg_se3 of a deformation node */
    std::shared_ptr<DualQuaternion<float>>& getTransformation();
    /* set transformation stored in a deformation node */
    void setTransformation(std::shared_ptr<DualQuaternion<float>> new_dg_se3);
    /* update transformation stored in a deformation nodes */
    void updateTransformation(boost::math::quaternion<float> real, boost::math::quaternion<float> dual);

    /* get radial basis weight of a vertex */
    float getRadialBasisWeight();
    /* get transformation weight for a vertex */
    float getTransformationWeight(pcl::PointXYZ vertexPosition);

private:
    pcl::PointXYZ dg_v;
    std::shared_ptr<DualQuaternion<float>> dg_se3;
    float dg_w;
};

/* DYNFU_NODE_HPP */
#endif
