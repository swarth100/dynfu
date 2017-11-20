#ifndef SOLVER_H
#define SOLVER_H

#include "ceres/ceres.h"

#include "warp_field.hpp"

class Solver {
public:
  Solver(ceres::Solver::Options options);
  ~Solver();

  template <typename T>
  void calculateWarpToLive(WarpField::WarpField warpField, T LiveFrame);

private:
  ceres::Solver::Options options;
};

#endif