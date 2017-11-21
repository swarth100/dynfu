#include <math.h>
#include <tgmath.h>

#include <dynfu/warp_field.hpp>

/* TODO: Add comment */
WarpField::WarpField() {}

/* TODO: Add comment */
WarpField::~WarpField() {}

/* TODO: Add comment */
void WarpField::init(std::shared_ptr<Frame> canonicalFrame) {
    // initialise all deformation nodes
}

/* TODO: Add comment */
void WarpField::warp() {
    // calculate DQB for all points
    // warps all points
}

/*
 * Returns a vector of all nodes in the warp field.
 */
std::vector<std::shared_ptr<Node>> WarpField::getNodes() {}
