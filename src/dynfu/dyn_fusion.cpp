#include <dynfu/dyn_fusion.hpp>

DynFusion::DynFusion() = default;
/* We initialise the dynamic fusion with the initals vertices and normals */
DynFusion::DynFusion(std::vector<cv::Vec3f> vertices, std::vector<cv::Vec3f> /* normals */) {
    /* Sample the deformation nodes */
    int steps = 50;
    std::vector<std::shared_ptr<Node>> deformationNodes;
    for (int i = 0; i < vertices.size(); i += steps) {
        auto dq = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
        deformationNodes.push_back(std::make_shared<Node>(vertices[i], dq, 0.f));
    }
    /* Initialise the warp field with the inital frames vertices */
    warpfield = std::make_shared<Warpfield>();
    warpfield->init(deformationNodes);

    for (auto node : deformationNodes) {
        node->setNeighbours(warpfield->findNeighbors(KNN, node));
    }
}

void DynFusion::init(kfusion::cuda::Cloud &vertices) {
    cv::Mat cloudHost;
    cloudHost.create(vertices.rows(), vertices.cols(), CV_32FC4);
    vertices.download(cloudHost.ptr<kfusion::Point>(), cloudHost.step);
    std::vector<cv::Vec3f> canonical(cloudHost.rows * cloudHost.cols);

    for (int y = 0; y < cloudHost.cols; ++y) {
        for (int x = 0; x < cloudHost.rows; ++x) {
            auto point = cloudHost.at<kfusion::Point>(x, y);
            if (!(std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))) {
                canonical[x + cloudHost.rows * y] = cv::Vec3f(point.x, point.y, point.z);
            }
        }
    }

    /* Sample the deformation nodes */
    int steps = 50;
    std::vector<std::shared_ptr<Node>> deformationNodes;
    for (int i = 0; i < canonical.size() - steps; i += steps) {
        auto dq = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
        deformationNodes.push_back(std::make_shared<Node>(canonical[i], dq, 0.f));
    }
    /* Initialise the warp field with the inital frames vertices */
    warpfield = std::make_shared<Warpfield>();
    warpfield->init(deformationNodes);

    for (auto node : deformationNodes) {
        node->setNeighbours(warpfield->findNeighbors(KNN, node));
    }
}

/* TODO: Add comment */
DynFusion::~DynFusion() = default;

/* TODO: Add comment */
void DynFusion::initCanonicalFrame() {}

/* TODO: Add comment */
void DynFusion::warpCanonicalToLive() {
    // query the solver passing to it the canonicalFrame, liveFrame, and
    // prevwarpField
}
