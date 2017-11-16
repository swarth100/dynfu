#include <iostream>
#include <kfusion/kinfu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace kfusion;

struct DynFuApp {
    DynFuApp(std::string filePath) : exit_(false), filePath_(filePath) {
        KinFuParams params = KinFuParams::default_params();
        kinfu_ = KinFu::Ptr(new KinFu(params));
    }

    void show_depth(const cv::Mat &depth) {
        cv::Mat display;
        // cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0 / 4000);
        cv::imshow("Depth", display);
    }

    void show_raycasted(KinFu &kinfu) {
        const int mode = 3;
        kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        cv::imshow("Scene", view_host_);
        cvWaitKey(100);
    }

    void take_cloud(KinFu &kinfu) {
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
    }

    void loadFiles(std::vector<cv::String> &depths, std::vector<cv::String> &images) {
        cv::glob(filePath_ + "/depth", depths);
        cv::glob(filePath_ + "/color", images);
        std::sort(depths.begin(), depths.end());
        std::sort(images.begin(), images.end());
    }

    bool execute() {
        KinFu &kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;
        cv::namedWindow("Image", cv::WINDOW_AUTOSIZE );
        cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE );
        cv::namedWindow("Scene", cv::WINDOW_AUTOSIZE );
        std::vector<cv::String> depths;
        std::vector<cv::String> images;
        loadFiles(depths, images);
        for (int i = 0; i < depths.size(); ++i) {
            auto depth = cv::imread(depths[i], CV_LOAD_IMAGE_ANYDEPTH);
            auto image = cv::imread(images[i], CV_LOAD_IMAGE_COLOR);
            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            {
             SampledScopeTime fps(time_ms);
            (void)fps;
             has_image = kinfu(depth_device_);
            }
            if (has_image) {
                show_raycasted(kinfu);
            }
            show_depth(depth);
            cv::imshow("Image", image);
            cv::imshow("Depth", depth);
        }
        return true;
    }

    std::string filePath_;
    bool exit_;
    KinFu::Ptr kinfu_;
    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    int device = 0;
    cuda::setDevice(device);
    cuda::printShortCudaDeviceInfo(device);

    if (cuda::checkIfPreFermiGPU(device)) {
        std::cout << std::endl
            << "Kinfu is not supported for pre-Fermi GPU "
            "architectures, and not built for them by "
            "default. Exiting..."
            << std::endl;
        return 1;
    }

    DynFuApp app(argv[1]);

    // executing
    try {
        app.execute();
    } catch (const std::bad_alloc & /*e*/) {
        std::cout << "Bad alloc" << std::endl;
    } catch (const std::exception & /*e*/) {
        std::cout << "Exception" << std::endl;
    }

    return 0;
}
