#include <dynfu/warp_field.hpp>

/* -------------------------------------------------------------------------- */
/* PUBLIC METHODS */

Warpfield::Warpfield() = default;

Warpfield::~Warpfield() = default;

void Warpfield::init(float epsilon, std::vector<std::shared_ptr<Node>> nodes) {
    /* init decimation density */
    this->epsilon = epsilon;
    /* initialise deformation nodes */
    this->nodes = nodes;

    /* hold deformation nodes' positions */
    std::vector<cv::Vec3f> deformationNodesPosition;
    for (auto node : this->nodes) {
        deformationNodesPosition.emplace_back(
            cv::Vec3f(node->getPosition().x, node->getPosition().y, node->getPosition().z));
    }

    /* initialise kd-tree */
    cloud      = std::make_shared<nanoflann::PointCloud>();
    cloud->pts = deformationNodesPosition;
    kdTree     = std::make_shared<kd_tree_t>(3, *cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdTree->buildIndex();
}

void Warpfield::addNode(std::shared_ptr<Node> newNode) { nodes.emplace_back(newNode); }

std::vector<std::shared_ptr<Node>> Warpfield::getNodes() { return this->nodes; }

pcl::PointCloud<pcl::PointXYZ>::Ptr Warpfield::getUnsupportedVertices(std::shared_ptr<dynfu::Frame> frame) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr unsupportedVertices(new pcl::PointCloud<pcl::PointXYZ>);

    for (auto vertex : frame->getVertices()) {
        std::vector<std::shared_ptr<Node>> neighbours = this->findNeighbors(KNN, vertex);

        float min = HUGE_VALF;

        for (auto neighbour : neighbours) {
            pcl::PointXYZ neighbourCoordinates = neighbour->getPosition();

            float dist = sqrt(pow(vertex.x - neighbourCoordinates.x, 2) + pow(vertex.y - neighbourCoordinates.y, 2) +
                              pow(vertex.z - neighbourCoordinates.z, 2));

            if (dist / neighbour->getRadialBasisWeight() <= min) {
                min = dist / neighbour->getRadialBasisWeight();
            }
        }

        if (min >= 1) {
            unsupportedVertices->push_back(vertex);
        }
    }

    std::cout << "no. of unsupported vertices: " << unsupportedVertices->size() << std::endl;

    return unsupportedVertices;
}

void Warpfield::update(std::shared_ptr<dynfu::Frame> frame) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr unsupportedVertices = getUnsupportedVertices(frame);
    pcl::PointCloud<pcl::PointXYZ>::Ptr unsupportedVerticesDownsampled(new pcl::PointCloud<pcl::PointXYZ>);

    /* voxel grid sampler */
    pcl::VoxelGrid<pcl::PointXYZ> sampler;
    sampler.setInputCloud(unsupportedVertices);
    sampler.setLeafSize(0.05, 0.05f, 0.05f);
    sampler.filter(*unsupportedVerticesDownsampled);

    std::cout << "no. of unsupported vertices after subsampling: " << unsupportedVerticesDownsampled->size()
              << std::endl;

    for (int i = 0; i < unsupportedVerticesDownsampled->size(); i++) {
        pcl::PointXYZ dg_v                            = (*unsupportedVerticesDownsampled)[i];
        std::shared_ptr<DualQuaternion<float>> dg_se3 = calcDQB(dg_v);
        float dg_w                                    = 2 * epsilon;

        std::shared_ptr<Node> newNode = std::make_shared<Node>(dg_v, dg_se3, dg_w);
        addNode(newNode);
    }

    /* re-initailise kd-tree */
    std::vector<cv::Vec3f> deformationNodesPosition;
    for (auto node : this->nodes) {
        deformationNodesPosition.emplace_back(
            cv::Vec3f(node->getPosition().x, node->getPosition().y, node->getPosition().z));
    }

    cloud->pts = deformationNodesPosition;
    kdTree     = std::make_shared<kd_tree_t>(3, *cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdTree->buildIndex();
}

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
