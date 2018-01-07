#include <dynfu/warp_field.hpp>

/* -------------------------------------------------------------------------- */
/* PUBLIC METHODS */

/* TODO: Add comment */
Warpfield::Warpfield() = default;

/* TODO: add comment */
Warpfield::Warpfield(const Warpfield& w) { init(w.nodes); }

/* TODO: Add comment */
Warpfield::~Warpfield() = default;

void Warpfield::init(std::vector<std::shared_ptr<Node>> nodes) {
    /* initialise deformation nodes */
    this->nodes = nodes;

    /* hold deformation nodes position */
    std::vector<cv::Vec3f> deformationNodesPosition;
    for (auto node : this->nodes) {
        deformationNodesPosition.push_back(node->getPosition());
    }

    cloud      = std::make_shared<nanoflann::PointCloud>();
    cloud->pts = deformationNodesPosition;
    kdTree     = std::make_shared<kd_tree_t>(3, *cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdTree->buildIndex();
}

std::vector<std::shared_ptr<Node>> Warpfield::getNodes() { return this->nodes; }

void Warpfield::addNode(std::shared_ptr<Node> newNode) { nodes.emplace_back(newNode); }

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

std::shared_ptr<dynfu::Frame> Warpfield::warpToCanonical(cv::Affine3f /* affineLiveToCanonical */,
                                                         std::shared_ptr<dynfu::Frame> liveFrame) {
    auto vertices = liveFrame->getVertices();
    auto normals  = liveFrame->getNormals();

    pcl::PointCloud<pcl::PointXYZ> warpedVertices;
    pcl::PointCloud<pcl::Normal> warpedNormals;

    for (int i = 0; i < vertices.size(); i++) {
        pcl::PointXYZ vertex = vertices[i];  // affineLiveToCanonical * vertices[i];
        pcl::Normal normal   = normals[i];   // affineLiveToCanonical * normals[i];

        auto transformation = calcDQB(vertex);

        vertex = pcl::PointXYZ(vertex.x - transformation->getTranslation()[0],
                               vertex.y - transformation->getTranslation()[1],
                               vertex.z - transformation->getTranslation()[2]);

        normal = pcl::Normal(normal.data_c[0] - transformation->getTranslation()[0],
                             normal.data_c[1] - transformation->getTranslation()[1],
                             normal.data_c[2] - transformation->getTranslation()[2]);

        warpedVertices.push_back(vertex);
        warpedNormals.push_back(normal);
    }

    return std::make_shared<dynfu::Frame>(0, warpedVertices, warpedNormals);
}

std::shared_ptr<dynfu::Frame> Warpfield::warpToLive(cv::Affine3f /* affineCanonicalToLive */,
                                                    std::shared_ptr<dynfu::Frame> canonicalFrame) {
    auto vertices = canonicalFrame->getVertices();
    auto normals  = canonicalFrame->getNormals();

    pcl::PointCloud<pcl::PointXYZ> warpedVertices;
    pcl::PointCloud<pcl::Normal> warpedNormals;

    for (int i = 0; i < vertices.size(); i++) {
        pcl::PointXYZ vertex = vertices[i];  // affineLiveToCanonical * vertices[i];
        pcl::Normal normal   = normals[i];   // affineLiveToCanonical * normals[i];

        auto transformation = calcDQB(vertex);

        vertex = pcl::PointXYZ(vertex.x + transformation->getTranslation()[0],
                               vertex.y + transformation->getTranslation()[1],
                               vertex.z + transformation->getTranslation()[2]);

        normal = pcl::Normal(normal.data_c[0] + transformation->getTranslation()[0],
                             normal.data_c[1] + transformation->getTranslation()[1],
                             normal.data_c[2] + transformation->getTranslation()[2]);

        warpedVertices.push_back(vertex);
        warpedNormals.push_back(normal);
    }

    return std::make_shared<dynfu::Frame>(0, warpedVertices, warpedNormals);
}

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

/* -------------------------------------------------------------------------- */
/* PRIVATE METHODS */

int Warpfield::getFrameNum() { return this->frameNum++; }
