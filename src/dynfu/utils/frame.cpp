#include <dynfu/utils/frame.hpp>

Frame(int id, std::vector<cv::Vec3f> vertices, std::vector<cv::Vec3f> normals) {
    this.id        = id;
    this->vertices = vertices;
    this->normals  = normals;
}

~Frame() {
    delete vertices;
    delete normals;
}

int getId() { return this.id; }

std::vector<cv::Vec3f> getVertices() { return this->vertices; }

std::vector<cv::Vec3f> getNormals() { return this->normals; }