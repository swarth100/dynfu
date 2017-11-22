#include <dynfu/dyn_fusion.hpp>

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

/* TODO: Add comment */
DynFusion::~DynFusion() = default;

/* TODO: Add comment */
void DynFusion::initCanonicalFrame() {}

/* TODO: Add comment */
void DynFusion::warpCanonicalToLive() {
    // query the solver passing to it the canonicalFrame, liveFrame, and
    // prevwarpField
}
