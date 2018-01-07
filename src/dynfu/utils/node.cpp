/* dynfu includes */
#include <dynfu/utils/node.hpp>

/* sys headers */
#include <cmath>

Node::Node(cv::Vec3f position, std::shared_ptr<DualQuaternion<float>> transformation, float radialBasisWeight) {
    this->dg_v   = position;
    this->dg_se3 = transformation;
    this->dg_w   = radialBasisWeight;
}

Node::~Node() = default;

cv::Vec3f Node::getPosition() { return dg_v; }

std::shared_ptr<DualQuaternion<float>>& Node::getTransformation() { return dg_se3; };

float Node::getRadialBasisWeight() { return dg_w; }

float Node::getTransformationWeight(pcl::PointXYZ vertexPosition) {
    auto position = this->getPosition();
    auto weight   = this->getRadialBasisWeight();
    auto dist     = pow(position[0] - vertexPosition.x, 2) + pow(position[1] - vertexPosition.y, 2) +
                pow(position[2] - vertexPosition.z, 2);

    return exp(-dist / (2 * pow(weight, 2)));
}

void Node::setTranslation(cv::Vec3f translation) {
    auto real    = dg_se3->getReal();
    this->dg_se3 = std::make_shared<DualQuaternion<float>>(real, translation);
}

void Node::setRotation(boost::math::quaternion<float> real) {
    auto dual    = dg_se3->getDual();
    this->dg_se3 = std::make_shared<DualQuaternion<float>>(real, dual);
}

void Node::setTransformation(std::shared_ptr<DualQuaternion<float>> transformation) { dg_se3 = transformation; }

void Node::setRadialBasisWeight(float newWeight) { dg_w = newWeight; }

void Node::updateTranslation(cv::Vec3f translation) {
    auto real = dg_se3->getReal();
    translation += dg_se3->getTranslation();
    this->dg_se3 = std::make_shared<DualQuaternion<float>>(real, translation);
}

void Node::updateRotation(cv::Vec3f eulerAngles) {
    DualQuaternion<float> q1(eulerAngles[0], eulerAngles[1], eulerAngles[2], 0.f, 0.f, 0.f);
    DualQuaternion<float> q2 = *(dg_se3);
    this->dg_se3             = std::make_shared<DualQuaternion<float>>(q1 * q2);
}
