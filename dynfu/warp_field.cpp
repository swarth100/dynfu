#include <math.h>
#include <tgmath.h>

#include <dynfu/warp_field.hpp>

Node::Node() {}

Node::~Node() {}

void Node::getNearestNeighbours() {}

// TODO: figure out x_c in the exponent
void Node::setWeight() {
    float exponent = 1.f;
    dg_w           = exp(exponent / (2.f * dg_w * dg_w));
}

WarpField::WarpField(std::vector<Node> *nodes) {}

WarpField::~WarpField() { delete nodes; }

std::vector<Node> *WarpField::getNodes() { return nodes; }
