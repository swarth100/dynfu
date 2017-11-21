#ifndef COST_FUNCTOR_HPP
#define COST_FUNCTOR_HPP

/* dynfu includes */
#include <dynfu/utils/frame.hpp>

class CostFunctor {
public:
    CostFunctor(Frame liveFrame, Frame canonicalFrame, WarpField warpField);

    template <typename T>
    bool operator()(Frame liveFrame, Frame canonicalFrame, T const* const* parameters, T* residuals);

private:
    T const* const* parameters;
};

#endif
