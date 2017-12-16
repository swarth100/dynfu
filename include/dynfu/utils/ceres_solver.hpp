#pragma once

/* dynfu includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>

#include <dynfu/dyn_fusion.hpp>
#include <dynfu/warp_field.hpp>

/* ceres includes */
#include <ceres/ceres.h>

/* sys headers */
#include <cmath>
#include <iostream>
#include <memory>

#define KNN 8

class Energy {
public:
    Energy(cv::Vec3f sourceVertex, cv::Vec3f liveVertex, std::vector<std::shared_ptr<Node>> liveVertexNeighbours) {
        this->sourceVertex         = sourceVertex;
        this->liveVertex           = liveVertex;
        this->liveVertexNeighbours = liveVertexNeighbours;
    }

    template <typename T>
    static T calcTransformationWeight(cv::Vec3f point1, cv::Vec3f point2, T radialBasisWeight) {
        T distance = T(cv::norm(point1 - point2));

        // if the distance between the node and the vertex is 0, return 1 for the transformation weight
        if (distance == T(0.0)) {
            return T(1.0);
        }

        return exp(-1.0 * distance * distance / (2.0 * radialBasisWeight * radialBasisWeight));
    }

    /*
     * calculates the residual of the linear system after applying weighted translation
     */
    template <typename T>
    bool operator()(T const* const* transformationParameters, T* residual) {
        T total_translation[3] = {T(0), T(0), T(0)};

        int i = 0;
        for (auto node : liveVertexNeighbours) {
            T weight = calcTransformationWeight(node->getPosition(), liveVertex, T(transformationParameters[i][0]));

            total_translation[0] += transformationParameters[i][1] * weight;
            total_translation[1] += transformationParameters[i][2] * weight;
            total_translation[2] += transformationParameters[i][3] * weight;
            i++;
        }

        residual[0] = T(liveVertex[0]) + total_translation[0] - T(sourceVertex[0]);
        residual[1] = T(liveVertex[1]) + total_translation[1] - T(sourceVertex[1]);
        residual[2] = T(liveVertex[2]) + total_translation[2] - T(sourceVertex[2]);

        return true;
    }

    /*
     * factory to hide the construction of the CostFunction object from
     * the client code
     */
    static ceres::CostFunction* Create(cv::Vec3f sourceVertex, cv::Vec3f liveVertex,
                                       std::vector<std::shared_ptr<Node>> liveVertexNeighbours) {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<Energy, 4 /* stride */>(
            new Energy(sourceVertex, liveVertex, liveVertexNeighbours));

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

    double const* const*
        transformationParameters;  // field to store the 3 components of the translation vector and the weight
};

/* class which allows to optimise the warpfield given the canonical and live frame */
class WarpProblem {
public:
    /* constructor; takes as argument options for the solver */
    WarpProblem(ceres::Solver::Options& options) { this->options = options; }

    /* destructor */
    ~WarpProblem() {}

    void optimiseWarpField(Warpfield warpfield, std::shared_ptr<dynfu::Frame> canonicalFrame,
                           std::shared_ptr<dynfu::Frame> liveFrame) {
        ceres::Problem problem;
        ceres::Solver::Summary summary;

        ceres::CostFunction* cost_function;
        std::vector<double*> values;

        int i = 0;
        for (auto vertex : liveFrame->getVertices()) {
            values.clear();
            auto neighbours = warpfield.findNeighbors(KNN, vertex);

            for (auto neighbour : neighbours) {
                values.emplace_back(neighbour->getParams());
            }

            cost_function = Energy::Create(vertex, canonicalFrame->getVertices()[i], neighbours);

            problem.AddResidualBlock(cost_function, NULL, values);
            i++;
        }

        Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";

        /* get parameter blocks */
        problem.GetParameterBlocks(&parameters);
    }

    std::vector<double*> getParameters() { return parameters; }

private:
    ceres::Solver::Options options;
    std::vector<double*> parameters;
};
