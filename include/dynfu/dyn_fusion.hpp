#ifndef DYNFU_DYNFUSION_HPP
#define DYNFU_DYNFUSION_HPP

#include <dynfu/utils/solver.hpp>
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
    std::shared_ptr<Warpfield> prevWarpfield;
    std::shared_ptr<Solver<double>> solver;
};

/* DYNFU_DYNFUSION_HPP */
#endif
