#include <iostream>
#include <kfusion/kinfu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace kfusion;

struct DynFuApp {
    DynFuApp(std::string filePath, bool visualizer) : exit_(false), filePath_(filePath), visualizer_(visualizer) {
        KinFuParams params = KinFuParams::default_params();
        kinfu_             = KinFu::Ptr(new KinFu(params));
    }

    void show_raycasted(KinFu *kinfu, int i) {
        const int mode = 3;
        (*kinfu).renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        if (visualizer_) {
            cv::imshow("Scene", view_host_);
            cvWaitKey(10);
        }
        std::string path = outPath_ + "/" + std::to_string(i) + ".png";
        cv::cvtColor(view_host_, view_host_, CV_BGR2GRAY);
        cv::imwrite(path, view_host_);
    }

    void take_cloud(KinFu *kinfu) {
        cuda::DeviceArray<Point> cloud = (*kinfu).tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, static_cast<int>(cloud.size()), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
    }

    void loadFiles(std::vector<cv::String> *depths, std::vector<cv::String> *images) {
        if (!boost::filesystem::exists(filePath_)) {
            std::cerr << "Error: Directory '" << filePath_ << "' does not exist. Exiting..." << std::endl;
            exit(EXIT_FAILURE);
        }

        if (!boost::filesystem::exists(filePath_ + "/depth") || !boost::filesystem::exists(filePath_ + "/color")) {
            std::cerr << "Error: Directory should contain 'color' and 'depth' directories. Exiting..." << std::endl;
            exit(EXIT_FAILURE);
        }

        cv::glob(filePath_ + "/depth", *depths);
        cv::glob(filePath_ + "/color", *images);
        std::sort((*depths).begin(), (*depths).end());
        std::sort((*images).begin(), (*images).end());
    }

    /* Create a new directory /out if not already present inside the input folder */
    void createOutputDirectory() {
        outPath_ = filePath_ + "/out";
        boost::filesystem::path dir(outPath_);
        if (boost::filesystem::create_directory(dir)) {
            std::cout << "Created output dir." << std::endl;
        }
    }

    bool execute() {
        KinFu &kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;
        if (visualizer_) {
            cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
            cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
            cv::namedWindow("Scene", cv::WINDOW_AUTOSIZE);
        }
        std::vector<cv::String> depths;
        std::vector<cv::String> images;
        loadFiles(&depths, &images);
        createOutputDirectory();
        for (int i = 0; i < depths.size(); ++i) {
            auto depth = cv::imread(depths[i], CV_LOAD_IMAGE_ANYDEPTH);
            auto image = cv::imread(images[i], CV_LOAD_IMAGE_COLOR);

            if (!image.data || !depth.data) {
                std::cerr << "Error: Image could not be read. Check for improper"
                          << " permissions, or invalid formats. Exiting..." << std::endl;
                exit(EXIT_FAILURE);
            }

            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            {
                SampledScopeTime fps(time_ms);
                (void) fps;
                has_image = kinfu(depth_device_);
            }
            if (has_image) {
                show_raycasted(&kinfu, i);
            }
            // show_depth(depth);
            if (visualizer_) {
                cv::imshow("Image", image);
                cv::imshow("Depth", depth);
            }
        }
        return true;
    }

    KinFu::Ptr kinfu_;
    std::string filePath_;
    std::string outPath_;
    bool exit_, visualizer_;
    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Parse the input flag and determine the file path and whether or not to enable visualizer
 * Any flags will be matched and the last argument which does not match the flag will be
 * treated as filepath
 */
void parseFlags(std::vector<std::string> args, std::string *filePath, bool *visualizer) {
    std::vector<std::string> flags = {"-h", "--help", "--enable-viz"};
    for (auto arg : args) {
        if (std::find(std::begin(flags), std::end(flags), arg) != std::end(flags)) {
            if (arg == "-h" || arg == "--help") {
                std::cout << "USAGE: dynamicfusion [OPTIONS] <file path>" << std::endl;
                std::cout << "\t--help -h:    Display help" << std::endl;
                std::cout << "\t--enable-viz: Enable visualizer" << std::endl;
                std::exit(EXIT_SUCCESS);
            }
            if (arg == "--enable-viz") {
                *visualizer = true;
            }
        } else {
            *filePath = arg;
        }
    }
}
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

    /* Program requires at least one argument - the path to the directory where the source files are */
    if (argc < 2) {
        return std::cerr << "Error: incorrect number of arguments. Please supply path to source data. Exiting..."
                         << std::endl,
               -1;
    }

    std::vector<std::string> args(argv + 1, argv + argc);
    std::string filePath;
    bool visualizer = false;
    parseFlags(args, &filePath, &visualizer);

    /* Disable the visualiser when running over SSH */
    if (visualizer && getenv("SSH_CLIENT")) {
        return std::cerr << "Error: cannot run visualiser while in SSH environment. Please run locally or disable "
                            "visualiser. Exiting..."
                         << std::endl,
               -1;
    }

    DynFuApp app(filePath, visualizer);

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
