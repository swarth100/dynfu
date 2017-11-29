#pragma once

/* dynfu includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>

#include <dynfu/dyn_fusion.hpp>
#include <dynfu/warp_field.hpp>

/* ceres includes */
#include <ceres/ceres.h>

/* sys headers */
#include <iostream>
#include <memory>

#define KNN 8

class Energy {
public:
    Energy(Warpfield warpfield, cv::Vec3f sourceVertex, cv::Vec3f liveVertex) {
        warpfield = warpfield;

        this->sourceVertex = sourceVertex;
        this->liveVertex   = liveVertex;

        std::cout << "--------------------- INIT ---------------------------" << std::endl;

        std::cout << "SOURCE VALUES: " << sourceVertex[0] << " " << sourceVertex[1] << " " << sourceVertex[2]
                  << std::endl;
        std::cout << "LIVE VALUES: " << liveVertex[0] << " " << liveVertex[1] << " " << liveVertex[2] << std::endl;

        std::cout << "--------------------- END INIT ---------------------------" << std::endl;

        /* TODO(rm3115) The warpfield is not warped, so the closest neighbours maybe wrong */
        liveVertexNeighbours = warpfield.findNeighbors(KNN, liveVertex);
    }

    /*
     * calculates the residual of the linear system after applying weighted translation
     */
    template <typename T>
    bool operator()(T const* const* transformationParameters, T* residual) {
        T total_translation[3] = {T(0), T(0), T(0)};
        for (int i = 0; i < liveVertexNeighbours.size(); ++i) {
            T temp[4] = {T(transformationParameters[i][0]), T(transformationParameters[i][1]),
                         T(transformationParameters[i][2]), T(transformationParameters[i][3])};
            /* In order of weight, x, y, z translation */
            auto neighborNode = liveVertexNeighbours[i];
            auto prev         = neighborNode->getParams();
            auto prevWeight   = DynFusion::getWeight(neighborNode, sourceVertex);

            cv::Vec3f position(temp[1], temp[2], temp[3]);
            auto currWeight = DynFusion::getWeight(position, sourceVertex);

            total_translation[0] += T(((prev[1] * prev[0]) + temp[1]) * temp[0]);
            total_translation[1] += T(((prev[2] * prev[0]) + temp[2]) * temp[0]);
            total_translation[2] += T(((prev[3] * prev[0]) + temp[3]) * temp[0]);
        }

        total_translation[0] /= 8;
        total_translation[1] /= 8;
        total_translation[2] /= 8;

        residual[0] = T(liveVertex[0]) + total_translation[0] - T(sourceVertex[0]);
        residual[1] = T(liveVertex[1]) + total_translation[1] - T(sourceVertex[1]);
        residual[2] = T(liveVertex[2]) + total_translation[2] - T(sourceVertex[2]);

        /* Debugging the Solver */
        /*std::cout << "Input Parameters: " << transformationParameters[0][0] << " " << transformationParameters[0][1]
                  << " " << transformationParameters[0][2] << " " << transformationParameters[0][3] << std::endl;
        std::cout << "SOURCE VALUES: " << sourceVertex[0] << " " << sourceVertex[1] << " " << sourceVertex[2]
                  << std::endl;
        std::cout << "LIVE VALUES: " << liveVertex[0] << " " << liveVertex[1] << " " << liveVertex[2] << std::endl;
        std::cout << "TRANSLATION VALUES: " << total_translation[0] << " " << total_translation[1] << " "
                  << total_translation[2] << std::endl;
        std::cout << "Solving for residuals: " << residual[0] << " " << residual[1] << " " << residual[2] << std::endl;
      */

        return true;
    }

    /*
     * factory to hide the construction of the CostFunction object from
     * the client code
     */
    static ceres::CostFunction* Create(Warpfield warpfield, cv::Vec3f sourceVertex, cv::Vec3f liveVertex) {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<Energy, 4 /* stride */>(
            new Energy(warpfield, sourceVertex, liveVertex));

        for (int i = 0; i < KNN; i++) {
            cost_function->AddParameterBlock(4);
        }

        cost_function->SetNumResiduals(3);  // dimensionality of the residual

        return cost_function;
    }

private:
    cv::Vec3f sourceVertex;
    cv::Vec3f liveVertex;

    std::vector<std::shared_ptr<Node>> liveVertexNeighbours;

    Warpfield warpfield;

    double const* const*
        transformationParameters;  // field to store the 3 components of the translation vector and the weight
};

/* class which allows to optimise the warpfield given the canonical and live frame */
class WarpProblem {
public:
    /* constructor; takes as argument options for the solver */
    WarpProblem(ceres::Solver::Options options) { options = options; }

    /* destructor */
    ~WarpProblem() {}

    void optimiseWarpField(Warpfield warpfield, std::shared_ptr<Frame> canonicalFrame,
                           std::shared_ptr<Frame> liveFrame) {
        ceres::Problem problem;
        ceres::Solver::Summary summary;

        ceres::CostFunction* cost_function;

        /* TODO(rm3115) Warp the canonical frame to live frame */
        int i = 0;
        for (auto vertex : liveFrame->getVertices()) {
            std::vector<double*> values;
            auto neighbours = warpfield.findNeighbors(8, vertex);

            for (auto neighbour : neighbours) {
                values.emplace_back(neighbour->getParams());
            }

            std::cout << values.size() << std::endl;
            std::cout << "Vertex coords: " << vertex[0] << " " << vertex[1] << " " << vertex[2] << std::endl;
            std::cout << "Canonical coords: " << canonicalFrame->getVertices()[i][0] << " "
                      << canonicalFrame->getVertices()[i][1] << " " << canonicalFrame->getVertices()[i][2] << std::endl;
            /* TODO(rm3115) We probably need to free the energy */
            cost_function = Energy::Create(warpfield, vertex, canonicalFrame->getVertices()[i]);
            problem.AddResidualBlock(cost_function, NULL, values);
            i++;
        }

        Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";

        /* get parameter blocks */
        problem.GetParameterBlocks(&parameters);

        for (auto parameter : parameters) {
            std::cout << parameter[0] << " " << parameter[1] << " " << parameter[2] << " " << parameter[3] << std::endl;
        }

        /* get IDs of residual blocks */
        /*
        problem.GetResidualBlocks(&blockIds);

        for (auto blockId : blockIds) {
            std::cout << "- - - - - -" << std::endl << "Block ID: " << blockId << std::endl;

            problem.GetParameterBlocksForResidualBlock(blockId, &residual_blocks);

            for (auto residual : residual_blocks) {
                std::cout << "Residual Data: " << residual[0] << " " << residual[1] << " " << residual[2] << std::endl;
            }
        }
        */
    }

private:
    ceres::Solver::Options options;
    std::vector<double*> parameters;

    std::vector<ceres::ResidualBlockId> blockIds;
    std::vector<double*> residual_blocks;
};
