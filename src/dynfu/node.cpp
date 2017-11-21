#include <dynfu/utils/node.hpp>

#include <c.math>

Node(cv::Vec3f position, std::shared_ptr<DualQuaternion<float>> transformation, float weight) {
    this.dg_v    = position;
    this->dg_se3 = transformation;
    this.dg_w    = weight;

    // TODO: init nearest neighbours
}

~Node() {}

std::shared_ptr<DualQuaternion<float>> getTransformation { return this->dg_se3; };

std::vector<std::shared_ptr<Node>> getNearestNeighbours { return this->nearestNeighbours; };

// TODO: finish once nearest neighbours have been initialised
void setWeight {
    float weight = 0;
    float sum    = 0;
    weight       = exp(-0.f / (2 * pow(sum, 2)));
}
