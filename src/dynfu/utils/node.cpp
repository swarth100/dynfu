#include <dynfu/utils/node.hpp>

/* sys headers */
#include <cmath>

Node::Node(cv::Vec3f position, std::shared_ptr<DualQuaternion<float>> transformation, float weight) {
    this->dg_v   = position;
    this->dg_se3 = transformation;
    this->dg_w   = weight;

    double temp1 = transformation->getTranslation()[0];
    double temp2 = transformation->getTranslation()[1];
    double temp3 = transformation->getTranslation()[2];

    this->params = new double[4]{weight, temp1, temp2, temp3};
}

Node::~Node() = default;

cv::Vec3f Node::getPosition() { return dg_v; }

double* Node::getParams() { return params; }

float Node::getWeight() { return dg_w; }

std::shared_ptr<DualQuaternion<float>>& Node::getTransformation() { return dg_se3; };

void Node::setTransformation(std::shared_ptr<DualQuaternion<float>> transformation) { dg_se3 = transformation; }

// TODO(dig15): finish once nearest neighbours have been initialised
void Node::setWeight() {
    float weight = 0;
    float sum    = 0;
    weight       = exp(-0.f / (2 * pow(sum, 2)));
}
