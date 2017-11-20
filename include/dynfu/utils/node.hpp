#ifndef DYNFU_NODE_HPP
#define DYNFU_NODE_HPP

/* Dynfu Includes */
#include <dynfu/utils/dual_quaternion.hpp>

/* OpenCV Includes */
#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

/* Sys Headers */
#include <memory>
#include <vector>

/*
 * Node of the warp field.
 * The state of the warp field at a given time list defined by the values of a
 * set of n deformation nodes.
 *
 * dg_v
 * Position of the node in space. This will be used when computing k-NN for
 * warping points.
 *
 * dg_se3
 * Transformation associated with a node.
 *
 * dg_w
 * Radial basis weight which controls the extent of transformation.
 */
class Node {
public:
    Node();
    ~Node();

    // DualQuaternion<float> getTransformation();
    void setWeight();

private:
    cv::Vec3f dg_v;
    std::shared_ptr<DualQuaternion<float>> dg_se3;
    float dg_w;

    std::vector<std::shared_ptr<Node>> *nearestNeighbours;
};

/* DYNFU_NODE_HPP */
#endif
