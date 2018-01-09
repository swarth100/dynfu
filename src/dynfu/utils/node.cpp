/* dynfu includes */
#include <dynfu/utils/node.hpp>

/* sys headers */
#include <cmath>

Node::Node(pcl::PointXYZ position, std::shared_ptr<DualQuaternion<float>> transformation, float radialBasisWeight) {
    this->dg_v   = position;
    this->dg_se3 = transformation;
    this->dg_w   = radialBasisWeight;
}

Node::~Node() = default;

pcl::PointXYZ Node::getPosition() { return dg_v; }

std::shared_ptr<DualQuaternion<float>>& Node::getTransformation() { return dg_se3; };

void Node::setTransformation(std::shared_ptr<DualQuaternion<float>> new_dg_se3) { this->dg_se3 = new_dg_se3; }

void Node::updateTransformation(boost::math::quaternion<float> real, boost::math::quaternion<float> dual) {
    auto prevTransformation = getTransformation();
    auto prevReal           = prevTransformation->getReal();
    auto prevDual           = prevTransformation->getDual();

    auto currReal = boost::math::quaternion<float>(1.f, 0.f, 0.f, 0.f);
    auto currDual = prevDual + dual;
    auto dq       = std::make_shared<DualQuaternion<float>>(currReal, currDual);

    this->dg_se3 = dq;
}

float Node::getRadialBasisWeight() { return dg_w; }

float Node::getTransformationWeight(pcl::PointXYZ vertexPosition) {
    auto position = this->getPosition();
    auto weight   = this->getRadialBasisWeight();
    auto distSq   = pow(position.x - vertexPosition.x, 2) + pow(position.y - vertexPosition.y, 2) +
                  pow(position.z - vertexPosition.z, 2);

    return exp(-distSq / (2 * pow(weight, 2)));
}
