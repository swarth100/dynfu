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
    Energy(Warpfield warpfield, cv::Vec3f sourceVertex, cv::Vec3f liveVertex) {
        warpfield = warpfield;

        this->sourceVertex = sourceVertex;
        this->liveVertex   = liveVertex;

        liveVertexNeighbours = warpfield.findNeighbors(KNN, liveVertex);
    }
    template <typename T>
    static T calcWeight(cv::Vec3f position, T weight, cv::Vec3f point) {
        cv::Vec<T, 3> distance_vec = cv::Vec<T, 3>(abs(T(position[0]) - T(point[0])), abs(T(position[1]) - T(point[1])),
                                                   abs(T(position[2]) - T(point[2])));
        T distance_norm = sqrt(abs(pow(distance_vec[0], 2.0) + pow(distance_vec[1], 2.0) + pow(distance_vec[2], 2.0)));

        if (distance_norm == T(0.0)) {
            return T(1.0);
        }
        return exp((-1.0 * pow(distance_norm, 2)) / (2.0 * pow(weight, 2)));
    }
    /*
     * calculates the residual of the linear system after applying weighted translation
     */
    template <typename T>
    bool operator()(T const* const* transformationParameters, T* residual) {
        T total_translation[3] = {T(0), T(0), T(0)};
        int i                  = 0;
        for (auto node : liveVertexNeighbours) {
            T temp[4] = {T(transformationParameters[i][0]), T(transformationParameters[i][1]),
                         T(transformationParameters[i][2]), T(transformationParameters[i][3])};

            T weight = calcWeight(node->getPosition(), temp[0], liveVertex);

            total_translation[0] += T(temp[1] * weight);
            total_translation[1] += T(temp[2] * weight);
            total_translation[2] += T(temp[3] * weight);
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
    WarpProblem(ceres::Solver::Options& options) { this->options = options; }

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

            cost_function = Energy::Create(warpfield, vertex, canonicalFrame->getVertices()[i]);
            problem.AddResidualBlock(cost_function, NULL, values);
            i++;
        }

        Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";

        /* get parameter blocks */
        problem.GetParameterBlocks(&parameters);

        // for (auto parameter : parameters) {
        // std::cout << parameter[0] << " " << parameter[1] << " " << parameter[2] << " " << parameter[3] << std::endl;
        //}
    }

    std::vector<double*> getParameters() { return parameters; }

private:
    ceres::Solver::Options options;
    std::vector<double*> parameters;

    std::vector<ceres::ResidualBlockId> blockIds;
    std::vector<double*> residual_blocks;
};
