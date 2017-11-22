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
<<<<<<< HEAD
    std::shared_ptr<Warpfield> prevWarpfield;
    std::shared_ptr<Solver> solver;
=======
    std::shared_ptr<WarpField> prevWarpField;
    // std::shared_ptr<Solver> solver;
>>>>>>> 6d1e276... /dynfu/: Fix Typos in frame, node, solver, and warp field implementations
};

/* DYNFU_DYNFUSION_HPP */
#endif
