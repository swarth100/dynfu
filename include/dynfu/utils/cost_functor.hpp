#ifndef COST_FUNCTOR_HPP
#define COST_FUNCTOR_HPP

/* System includes */
#include <memory>

/* dynfu includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/warp_field.hpp>

template <class T>
class CostFunctor {
public:
    CostFunctor(std::shared_ptr<Warpfield> warpfield, std::shared_ptr<Frame> canonicalFrame, std::shared_ptr<Frame> liveFrame) {
        auto nodes = warpfield->getNodes();
        int i = 0;
        for (auto node : nodes) {
            auto nearestNeighbours = node->getNearestNeighbours();
            for (auto nearestNeighbour : nearestNeighbours) {
                auto transformation       = nearestNeighbour->getTransformation();
                auto transformationWeight = nearestNeighbour->getWeight();

                this->parameters[i][0] += transformation->getTranslation()[0] * transformationWeight;
                this->parameters[i][1] += transformation->getTranslation()[1] * transformationWeight;
                this->parameters[i][2] += transformation->getTranslation()[2] * transformationWeight;

                this->parameters[i][3] += transformation->getRotation().R_component_2() * transformationWeight;
                this->parameters[i][4] += transformation->getRotation().R_component_3() * transformationWeight;
                this->parameters[i][5] += transformation->getRotation().R_component_4() * transformationWeight;
            }
            i++;
        }
    }

    bool operator()(std::shared_ptr<Frame> canonicalFrame, std::shared_ptr<Frame> liveFrame, T const* const* parameters,
                    T* residuals) {
        residuals[0] = T(liveFrame->getVertices()[0] - canonicalFrame->getVertices()[0]) - parameters[0] - parameters[3];
        residuals[1] = T(liveFrame->getVertices()[1] - canonicalFrame->getVertices()[1]) - parameters[1] - parameters[4];
        residuals[2] = T(liveFrame->getVertices()[2] - canonicalFrame->getVertices()[2]) - parameters[2] - parameters[5];

        return true;
    }

    T const* const* getParameters() { return parameters; }
private:
    T const* const* parameters;
};

#endif
