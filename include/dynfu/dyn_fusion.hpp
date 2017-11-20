#ifndef DYNFU_DYNFUSION_HPP
#define DYNFU_DYNFUSION_HPP

#include <dynfu/solver.hpp>
#include <dynfu/utils/frame.hpp>
#include <dynfu/warp_field.hpp>

/* */
class DynFusion {
public:
    DynFusion();
    ~DynFusion();

    void initCanonicalFrame();
    // void updateCanonicalFrame();
    void warpCanonicalToLive();

private:
    std::shared_ptr<Frame> canonicalFrame;
    std::shared_ptr<Frame> liveFrame;
    std::shared_ptr<Frame> canonicalWarpedToLive;
    std::shared_ptr<WarpField> prevWarpField;
    std::shared_ptr<Solver> solver;
};

/* DYNFU_DYNFUSION_HPP */
#endif
