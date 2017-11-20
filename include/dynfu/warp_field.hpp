#ifndef WARP_FIELD_HPP
#define WARP_FIELD_HPP

#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>

/*
 * Warp field.
 */
class WarpField {
public:
    WarpField();
    ~WarpField();

    void init(Frame canonicalFrame) {
        // initialise all deformation nodes
    }

    void warp() {
        // calculate DQB for all points
        // warps all points
    }

    /*
     * Returns a vector of all nodes in the warp field.
     */
    std::vector<Node> *getNodes();

private:
    std::vector<Node> *nodes;
};

#endif
