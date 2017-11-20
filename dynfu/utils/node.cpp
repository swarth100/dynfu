#include <dynfu/utils/node.hpp>

/* TODO: Add comment */
Node::Node() {}

/* TODO: Add comment */
Node::~Node() {}

/* TODO: Add comment */
void Node::setWeight() {
    float exponent = 1.f;
    dg_w           = exp(exponent / (2.f * dg_w * dg_w));
}
