#include <ceres/ceres.h>

#include <dynfu/utils/frame.hpp>
#include <dynfu/warp_field.hpp>

struct CostFunctor {
    bool operator()(Warpfield warpfield, std::shared_ptr<Frame> canonicalFrame, std::shared_ptr<Frame> liveFrame,
                    float* residual) {
        auto nodes      = warpfield.getNodes();
        auto parameters = warpfield.getParameters();

        int i = 0;
        for (auto node : nodes) {
            auto nearestNeighbours = node->getNearestNeighbours();
            for (auto nearestNeighbour : nearestNeighbours) {
                auto transformation       = nearestNeighbour->getTransformation();
                auto transformationWeight = nearestNeighbour->getWeight();

                parameters[i][0] += transformation->getTranslation()[0] * transformationWeight;
                parameters[i][1] += transformation->getTranslation()[1] * transformationWeight;
                parameters[i][2] += transformation->getTranslation()[2] * transformationWeight;

                parameters[i][3] += transformation->getRotation().R_component_2() * transformationWeight;
                parameters[i][4] += transformation->getRotation().R_component_3() * transformationWeight;
                parameters[i][5] += transformation->getRotation().R_component_4() * transformationWeight;
            }
            i++;
        }

        residual[0] = liveFrame->getVertices()[0][0] - canonicalFrame->getVertices()[0][0] -
                      parameters[0][0];  // - parameters[3];
        residual[1] = liveFrame->getVertices()[0][1] - canonicalFrame->getVertices()[0][1] -
                      parameters[0][1];  // - parameters[4];
        residual[2] = liveFrame->getVertices()[0][2] - canonicalFrame->getVertices()[0][2] -
                      parameters[0][2];  // - parameters[5];

        return true;
    }
};
