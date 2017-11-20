#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

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
    // DualQuaternion<float> dg_se3;
    float dg_w;

    std::vector<Node> *nearestNeighbours;
};
