#include <dynfu/utils/node.hpp>

/* sys headers */
#include <cmath>

Node::Node(cv::Vec3f position, std::shared_ptr<DualQuaternion<float>> transformation, float weight) {
    this->dg_v    = position;
    this->dg_se3 = transformation;
    this->dg_w    = weight;

    // TODO: init nearest neighbours
}

Node::~Node() {}

const std::shared_ptr<DualQuaternion<float>>&  Node::getTransformation() { return dg_se3; };

const std::vector<std::shared_ptr<Node>>& Node::getNearestNeighbours() { return nearestNeighbours; };

// TODO: finish once nearest neighbours have been initialised
void setWeight() {
    float weight = 0;
    float sum    = 0;
    weight       = exp(-0.f / (2 * pow(sum, 2)));
}
