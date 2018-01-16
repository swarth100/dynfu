#include <dynfu/warp_field.hpp>

/* -------------------------------------------------------------------------- */
/* PUBLIC METHODS */

Warpfield::Warpfield() = default;

Warpfield::Warpfield(const Warpfield& w) { init(w.nodes); }

Warpfield::~Warpfield() = default;

void Warpfield::init(std::vector<std::shared_ptr<Node>> nodes) {
    /* initialise deformation nodes */
    this->nodes = nodes;

    /* hold deformation nodes' positions */
    std::vector<cv::Vec3f> deformationNodesPosition;
    for (auto node : this->nodes) {
        deformationNodesPosition.push_back(
            cv::Vec3f(node->getPosition().x, node->getPosition().y, node->getPosition().z));
    }

    /* initialise kd-tree */
    cloud      = std::make_shared<nanoflann::PointCloud>();
    cloud->pts = deformationNodesPosition;
    kdTree     = std::make_shared<kd_tree_t>(3, *cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdTree->buildIndex();
}

std::vector<std::shared_ptr<Node>> Warpfield::getNodes() { return this->nodes; }

void Warpfield::addNode(std::shared_ptr<Node> newNode) { nodes.emplace_back(newNode); }

std::shared_ptr<kd_tree_t> Warpfield::getKdTree() { return this->kdTree; }

std::vector<std::shared_ptr<Node>> Warpfield::findNeighbors(int numNeighbor, pcl::PointXYZ vertex) {
    auto retIndex = findNeighborsIndex(numNeighbor, vertex);

    std::vector<std::shared_ptr<Node>> neighborNodes;

    for (auto index : retIndex) {
        neighborNodes.push_back(nodes[index]);
    }

    return neighborNodes;
}

std::vector<size_t> Warpfield::findNeighborsIndex(int numNeighbor, pcl::PointXYZ vertex) {
    /* not used, ignores the distance to the nodes for now */
    std::vector<float> outDistSqr(numNeighbor);
    std::vector<size_t> retIndex(numNeighbor);

    /* unpack Vec3f into vector */
    std::vector<float> query = {vertex.x, vertex.y, vertex.z};
    int n                    = kdTree->knnSearch(&query[0], numNeighbor, &retIndex[0], &outDistSqr[0]);
    retIndex.resize(n);

    return retIndex;
}

/* calculate DQB */
/* get dg_se3 from each of the nodes, multiply it by the transformation weight, and sum */
/* before returning, normalise the dual quaternion */
std::shared_ptr<DualQuaternion<float>> Warpfield::calcDQB(pcl::PointXYZ point) {
    /* from the warp field get the 8 nearest deformation nodes */
    auto nearestNeighbors = this->findNeighbors(KNN, point);

    /* for each of the nodes, get the distance between the node and the point */
    /* apply the formula to get w(x) */
    DualQuaternion<float> transformationSum(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

    for (auto node : nearestNeighbors) {
        float nodeWeight = node->getTransformationWeight(point);

        DualQuaternion<float> dg_se3                  = *node->getTransformation();
        DualQuaternion<float> weighted_transformation = dg_se3 * nodeWeight;

        transformationSum *= weighted_transformation;
    }

    /* normalise the sum */
    DualQuaternion<float> dual_quaternion_blending = transformationSum.normalize();

    return std::make_shared<DualQuaternion<float>>(dual_quaternion_blending);
}

std::shared_ptr<dynfu::Frame> Warpfield::warpToLive(std::shared_ptr<dynfu::Frame> canonicalFrame) {
    pcl::PointCloud<pcl::PointXYZ> vertices = canonicalFrame->getVertices();
    pcl::PointCloud<pcl::Normal> normals    = canonicalFrame->getNormals();

    pcl::PointCloud<pcl::PointXYZ> warpedVertices;
    pcl::PointCloud<pcl::Normal> warpedNormals;

    for (int i = 0; i < vertices.size(); i++) {
        pcl::PointXYZ vertex = vertices[i];
        pcl::Normal normal   = normals[i];

        std::shared_ptr<DualQuaternion<float>> transformation = calcDQB(vertex);

        pcl::PointXYZ vertexNew = transformation->transformVertex(vertex);
        pcl::Normal normalNew   = transformation->transformNormal(normal);

        warpedVertices.push_back(vertexNew);
        warpedNormals.push_back(normalNew);
    }

    return std::make_shared<dynfu::Frame>(0, warpedVertices, warpedNormals);
}

/* -------------------------------------------------------------------------- */
/* PRIVATE METHODS */

int Warpfield::getFrameNum() { return this->frameNum++; }
