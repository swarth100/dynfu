#ifndef WARP_FIELD_HPP
#define WARP_FIELD_HPP

#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

#define KNN 8

class DynFusion {
public:
  void initCanonicalFrame();
  // void updateCanonicalFrame();
  void warpCanonicalToLive() {
    // query the solver passing to it the canonicalFrame, liveFrame, and
    // prevWarpField
  }

private:
  Frame canonicalFrame;
  Frame liveFrame;
  Frame canonicalWarpedToLive;
  WarpField prevWarpField;
  Solver solver;
};

class Frame {
public:
  Frame(void *addIt); // TODO figure out
  ~Frame();

  void getId();
  void getVertices();
  void getNormals();

private:
  int id;
  std::vector<cv::Vec3f> vertices;
  std::vector<cv::Vec3f> normals;
};

/*
 * Node of the warp field.
 * The state of the warp field at a given time list defined by the values of a
 * set of n deformation nodes.
 *
 * dg_v
 * Position of the node in space. This will be used when computing k-NN for
 * warping points.
 *
 * dg_se3
 * Transformation associated with a node.
 *
 * dg_w
 * Radial basis weight which controls the extent of transformation.
 */
class Node {
public:
  Node();
  ~Node();

  // DualQuaternion<float> getTransformation();
  void setWeight();

private:
  cv::Vec3f dg_v;
  // DualQuaternion<float> dg_se3;
  float dg_w;

  std::vector<Node> *nearestNeighbours;
};

/*
 * Warp field.
 */
class WarpField {
public:
  WarpField();
  ~WarpField();

  void init(Frame canonicalFrame) {
    // initialise all deformation nodes
  }

  void warp() {
    // calculate DQB for all points
    // warps all points
  }

  /*
   * Returns a vector of all nodes in the warp field.
   */
  std::vector<Node> *getNodes();

private:
  std::vector<Node> *nodes;
};

#endif