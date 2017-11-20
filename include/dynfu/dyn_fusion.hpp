#include <dynfu/solver.hpp>
#include <dynfu/utils/frame.hpp>
#include <dynfu/warp_field.hpp>

/* */
class DynFusion {
public:
    void initCanonicalFrame();
    // void updateCanonicalFrame();
    void warpCanonicalToLive() {
        // query the solver passing to it the canonicalFrame, liveFrame, and
        // prevWarpField
    }

private:
    Frame canonicalFrame;
    Frame liveFrame;
    Frame canonicalWarpedToLive;
    WarpField prevWarpField;
    Solver solver;
};
