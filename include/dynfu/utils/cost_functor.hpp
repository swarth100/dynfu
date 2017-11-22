#ifndef COST_FUNCTOR_HPP
#define COST_FUNCTOR_HPP

/* System includes */
#include <memory>

/* dynfu includes */

template<typename T>
class CostFunctor {
public:
    CostFunctor(std::shared<Frame> liveFrame, std::shared<Frame> canonicalFrame, std::shared<Warpfield> warpfield);

    template <typename T>
    bool operator()(std::shared<Frame> canonicalFrame, std::shared<Frame> liveFrame, T const* const* parameters, T* residuals);

private:
    T const* const* parameters;
};

#endif
