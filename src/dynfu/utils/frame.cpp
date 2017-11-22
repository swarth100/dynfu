#include <dynfu/utils/frame.hpp>

Frame::Frame(int id, std::vector<cv::Vec3f> vertices, std::vector<cv::Vec3f> normals) {
    this->id        = id;
    this->vertices = vertices;
    this->normals  = normals;
}

Frame::~Frame() = default;

int Frame::getId() { return this->id; }

std::vector<cv::Vec3f> Frame::getVertices() { return this->vertices; }

std::vector<cv::Vec3f> Frame::getNormals() { return this->normals; }
