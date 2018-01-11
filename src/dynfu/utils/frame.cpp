#include <dynfu/utils/frame.hpp>

dynfu::Frame::Frame(int id, std::vector<cv::Vec3f> vertices, std::vector<cv::Vec3f> normals) {
    this->id       = id;
    this->vertices = vertices;
    this->normals  = normals;
}

dynfu::Frame::~Frame() = default;

int dynfu::Frame::getId() { return this->id; }

std::vector<cv::Vec3f>& dynfu::Frame::getVertices() { return this->vertices; }

std::vector<cv::Vec3f>& dynfu::Frame::getNormals() { return this->normals; }
