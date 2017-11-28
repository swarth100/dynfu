#pragma once

/* dynfu includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>

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

        sourceVertex = sourceVertex;
        liveVertex   = liveVertex;

        liveVertexNeighbours = warpfield.findNeighbors(KNN, liveVertex);
    }

    /*
     * calculates the residual of the linear system after applying weighted translation
     */
    template <typename T>
    bool operator()(T const* const* transformationParameters, T* residual) {
        T total_translation[3] = {T(0), T(0), T(0)};

        for (auto node : liveVertexNeighbours) {
            T temp[4] = {T(*transformationParameters[0]), T(*transformationParameters[1]),
                         T(*transformationParameters[2]), T(*transformationParameters[3])};

            total_translation[0] += T(sourceVertex[0]) + T(temp[1] * temp[0]);
            total_translation[1] += T(sourceVertex[1]) + T(temp[2] * temp[0]);
            total_translation[2] += T(sourceVertex[2]) + T(temp[3] * temp[0]);
        }

        residual[0] = T(liveVertex[0]) - total_translation[0];
        residual[1] = T(liveVertex[1]) - total_translation[1];
        residual[2] = T(liveVertex[2]) - total_translation[2];

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

        int i = 0;
        for (auto vertex : liveFrame->getVertices()) {
            std::vector<double*> values;
            auto neighbours = warpfield.findNeighbors(8, vertex);

            for (auto neighbour : neighbours) {
                values.emplace_back(neighbour->getParams());
            }

            std::cout << values.size() << std::endl;

            cost_function = Energy::Create(warpfield, vertex, canonicalFrame->getVertices()[i]);
            problem.AddResidualBlock(cost_function, NULL, values);

            i++;
        }

        Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";

        problem.GetParameterBlocks(&parameters);

        for (auto parameter : parameters) {
            std::cout << *parameter << std::endl;
        }
    }

private:
    ceres::Solver::Options options;
    std::vector<double*> parameters;
};
