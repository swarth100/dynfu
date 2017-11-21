#include <dynfu/utils/cost_functor.hpp>

CostFunctor(Frame liveFrame, Frame canonicalFrame, WarpField warpField) {
    auto nodes = warpField.getNodes();

    for (std::size_t i = 0; i != (sizeof nodes / sizeof *nodes); i++) {
        auto nearestNeighbours = nodes[i].getNearestNeighbours();

        for (std::size_t k = 0; k != (sizeof nearestNeighbours / sizeof *nearestNeighbours); k++) {
            auto transformation       = nearestNeighbours[k].getTransformation();
            auto transformationWeight = nearestNeighbours[k].getWeight();

            this->parameters[i][0] += transformation.getTranslation()[0] * transformationWeight;
            this->parameters[i][1] += transformation.getTranslation()[1] * transformationWeight;
            this->parameters[i][2] += transformation.getTranslation()[2] * transformationWeight;

            this->parameters[i][3] += transformation.getRotation()[0] * transformationWeight;
            this->parameters[i][4] += transformation.getRotation()[1] * transformationWeight;
            this->parameters[i][5] += transformation.getRotation()[2] * transformationWeight;
        }
    }
}

template <typename T>
bool operator()(Frame liveFrame, Frame canonicalFrame, T const* const* parameters, T* residuals) const {
    residuals[0] = T(liveFrame.getVertices()[0] - canonicalFrame.getVertices()[0]) - parameters[0] - parameters[3];
    residuals[1] = T(liveFrame.getVertices()[1] - canonicalFrame.getVertices()[1]) - parameters[1] - parameters[4];
    residuals[2] = T(liveFrame.getVertices()[2] - canonicalFrame.getVertices()[2]) - parameters[2] - parameters[5];

    return true;
}