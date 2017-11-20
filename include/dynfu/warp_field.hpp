#ifndef DYNFU_WARP_FIELD_HPP
#define DYNFU_WARP_FIELD_HPP

/* Dynfu Includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>

/* */
class WarpField {
public:
    WarpField();
    ~WarpField();

    void init(std::shared_ptr<Frame> canonicalFrame);

    void warp();

    /*
     * Returns a vector of all nodes in the warp field.
     */
    std::vector<std::shared_ptr<Node>> getNodes();

private:
    std::vector<std::shared_ptr<Node>> nodes;
    std::shared_ptr<Frame> canonicalFrame;
};

/* DYNFU_WARP_FIELD_HPP */
#endif
