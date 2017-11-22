#ifndef COST_FUNCTOR_HPP
#define COST_FUNCTOR_HPP

/* dynfu includes */

template<typename T>
class CostFunctor {
public:
    CostFunctor(Frame liveFrame, Frame canonicalFrame, WarpField::WarpField warpField);

    template <typename T>
    bool operator()(Frame canonicalFrame, Frame liveFrame, T const* const* parameters, T* residuals);

private:
    T const* const* parameters;
};

#endif
