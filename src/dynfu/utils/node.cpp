#include <dynfu/utils/node.hpp>

/* sys headers */
#include <cmath>

Node::Node(cv::Vec3f position, std::shared_ptr<DualQuaternion<float>> transformation, float radialBasisWeight) {
    this->dg_v   = position;
    this->dg_se3 = transformation;
    this->dg_w   = radialBasisWeight;

    double temp1 = transformation->getTranslation()[0];
    double temp2 = transformation->getTranslation()[1];
    double temp3 = transformation->getTranslation()[2];

    this->params = new double[4]{radialBasisWeight, temp1, temp2, temp3};
}

Node::~Node() = default;

cv::Vec3f Node::getPosition() { return dg_v; }

std::shared_ptr<DualQuaternion<float>>& Node::getTransformation() { return dg_se3; };

float Node::getRadialBasisWeight() { return dg_w; }

double* Node::getParams() { return params; }

void Node::setTranslation(cv::Vec3f translation) {
    auto real = dg_se3->getReal();
    dg_se3    = std::make_shared<DualQuaternion<float>>(real, translation);
}

void Node::setRotation(boost::math::quaternion<float> real) {
    auto dual = dg_se3->getDual();
    dg_se3    = std::make_shared<DualQuaternion<float>>(real, dual);
}

void Node::setTransformation(std::shared_ptr<DualQuaternion<float>> transformation) { dg_se3 = transformation; }

void Node::setRadialBasisWeight(float newWeight) { dg_w = newWeight; }
