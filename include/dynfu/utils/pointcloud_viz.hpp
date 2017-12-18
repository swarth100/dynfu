#pragma once
/* sys headers */
#include <iostream>
#include <memory>

/* viz headers */
#include <opencv2/core/core.hpp>
#include <opencv2/viz.hpp>

class PointCloudViz {
private:
    std::shared_ptr<cv::viz::Viz3d> viewer;

public:
    PointCloudViz();
    std::shared_ptr<cv::viz::Viz3d> getViewer();
    cv::Mat vecToMat(std::vector<cv::Vec3f> vec);
    cv::viz::WCloud matToCloud(cv::Mat mat);
};
