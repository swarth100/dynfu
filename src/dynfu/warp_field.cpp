#include <dynfu/warp_field.hpp>

/* -------------------------------------------------------------------------- */
/* PUBLIC METHODS */

/* TODO: Add comment */
Warpfield::Warpfield() = default;

Warpfield::Warpfield(const Warpfield& w) { init(w.nodes); }

/* TODO: Add comment */
Warpfield::~Warpfield() = default;

/* TODO: Add comment */
void Warpfield::init(std::vector<std::shared_ptr<Node>> nodes) {
    // initialise all deformation nodes
    this->nodes = nodes;

    /* Hold deformation nodes position */
    std::vector<cv::Vec3f> deformationNodesPosition;

    for (auto node : this->nodes) {
        deformationNodesPosition.push_back(node->getPosition());
    }

    cloud      = std::make_shared<nanoflann::PointCloud>();
    cloud->pts = deformationNodesPosition;
    kdTree     = std::make_shared<kd_tree_t>(3, *cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdTree->buildIndex();
}

/*
 * return a vector of all nodes in the warp field
 */
std::vector<std::shared_ptr<Node>> Warpfield::getNodes() { return this->nodes; }

void Warpfield::addNode(std::shared_ptr<Node> newNode) { nodes.emplace_back(newNode); }

/* Calculate Dual Quaternion Blending */
/* Get the dg_se3 from each of the nodes, time it by the weight and calculate the sum */
/* Before returning, normalise the dual quaternion */
std::shared_ptr<DualQuaternion<float>> Warpfield::calcDQB(cv::Vec3f point) {
    /* From the warp field get the k (8) closest points */
    auto nearestNeighbors = this->findNeighbors(KNN, point);
    /* Then for each of the Nodes compare the distance between the vector of the Node and the point */
    /* Apply the formula to get w(x) */
    DualQuaternion<float> transformationSum(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    for (auto node : nearestNeighbors) {
        float nodeWeight = node->getTransformationWeight(point);

        DualQuaternion<float> dg_se3                  = *node->getTransformation();
        DualQuaternion<float> weighted_transformation = dg_se3 * nodeWeight;

        transformationSum += weighted_transformation;
    }
    /*Normalise the sum */
    DualQuaternion<float> dual_quaternion_blending = transformationSum.normalize();

    return std::make_shared<DualQuaternion<float>>(dual_quaternion_blending);
}

std::shared_ptr<dynfu::Frame> Warpfield::warpToCanonical(std::shared_ptr<dynfu::Frame> liveFrame) {
    auto vertices = liveFrame->getVertices();
    auto normals  = liveFrame->getNormals();

    std::vector<cv::Vec3f> warpedVertices;
    std::vector<cv::Vec3f> warpedNormals;

    for (int i = 0; i < vertices.size(); i++) {
        cv::Vec3f vertex      = vertices[i];
        cv::Vec3f normal      = normals[i];
        auto transformation   = calcDQB(vertex);
        auto totalTranslation = transformation->getTranslation();
        vertex -= totalTranslation;
        normal -= totalTranslation;
        warpedVertices.emplace_back(vertex);
        warpedNormals.emplace_back(normal);
    }
    return std::make_shared<dynfu::Frame>(0, warpedVertices, warpedNormals);
}

std::shared_ptr<dynfu::Frame> Warpfield::warpToLive(std::shared_ptr<dynfu::Frame> canonicalFrame) {
    auto vertices = canonicalFrame->getVertices();
    auto normals  = canonicalFrame->getNormals();

    std::vector<cv::Vec3f> warpedVertices;
    std::vector<cv::Vec3f> warpedNormals;

    for (int i = 0; i < vertices.size(); i++) {
        cv::Vec3f vertex      = vertices[i];
        cv::Vec3f normal      = normals[i];
        auto transformation   = calcDQB(vertex);
        auto totalTranslation = transformation->getTranslation();
        vertex += totalTranslation;
        normal += totalTranslation;
        warpedVertices.emplace_back(vertex);
        warpedNormals.emplace_back(normal);
    }
    return std::make_shared<dynfu::Frame>(0, warpedVertices, warpedNormals);
}

/* Find the nodes of k closest neighbour for the given point */
std::vector<std::shared_ptr<Node>> Warpfield::findNeighbors(int numNeighbor, cv::Vec3f vertex) {
    auto retIndex = findNeighborsIndex(numNeighbor, vertex);
    std::vector<std::shared_ptr<Node>> neighborNodes;
    for (auto index : retIndex) {
        neighborNodes.push_back(nodes[index]);
    }
    return neighborNodes;
}

/* Find the nodes index of k closest neighbour for the given point */
std::vector<size_t> Warpfield::findNeighborsIndex(int numNeighbor, cv::Vec3f vertex) {
    /* Not used, ignores the distance to the nodes for now */
    std::vector<float> outDistSqr(numNeighbor);
    std::vector<size_t> retIndex(numNeighbor);
    /* Unpack the Vec3f into vector */
    std::vector<float> query = {vertex[0], vertex[1], vertex[2]};
    int n                    = kdTree->knnSearch(&query[0], numNeighbor, &retIndex[0], &outDistSqr[0]);
    retIndex.resize(n);
    return retIndex;
}

/* -------------------------------------------------------------------------- */
/* PRIVATE METHODS */

/* */
int Warpfield::getFrameNum() { return this->frameNum++; }
