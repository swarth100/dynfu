FROM nvidia/cuda:8.0-devel
ARG CUDA_GENERATION=Auto

RUN apt-get update && apt-get install -y cmake libvtk5-dev pkg-config libgtk2.0-dev

# Install OpenCV
ADD https://github.com/opencv/opencv/archive/3.2.0.tar.gz .
RUN tar xzf 3.2.0.tar.gz
RUN rm 3.2.0.tar.gz
WORKDIR opencv-3.2.0
RUN rm -rf platforms/android platforms/ios platforms/maven platforms/osx samples/*
RUN mkdir build
WORKDIR build
RUN cmake -D BUILD_DOCS=OFF \
          -D BUILD_PACKAGE=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_WITH_DEBUG_INFO=OFF \
          -D BUILD_opencv_apps=OFF \
          -D BUILD_opencv_calib3d=ON \
          -D BUILD_opencv_core=ON \
          -D BUILD_opencv_features2d=ON \
          -D BUILD_opencv_flann=ON \
          -D BUILD_opencv_highgui=ON \
          -D BUILD_opencv_imgcodecs=ON \
          -D BUILD_opencv_imgproc=ON \
          -D BUILD_opencv_ml=ON \
          -D BUILD_opencv_objdetect=OFF \
          -D BUILD_opencv_photo=OFF \
          -D BUILD_opencv_shape=OFF \
          -D BUILD_opencv_stitching=OFF \
          -D BUILD_opencv_superres=OFF \
          -D BUILD_opencv_ts=OFF \
          -D BUILD_opencv_video=OFF \
          -D BUILD_opencv_videoio=OFF \
          -D BUILD_opencv_videostab=OFF \
          -D BUILD_opencv_viz=ON \
          -D BUILD_opencv_video=OFF \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CUDA_GENERATION=${CUDA_GENERATION:-Auto} \
          -D WITH_VTK=ON \
    ..
RUN make -j`nproc` install
WORKDIR ../..
RUN rm -rf opencv-3.2.0

RUN apt-get install -y git zip clang libsuitesparse-dev liblapack-dev libblas-dev libeigen3-dev libgoogle-glog-dev

# Install ceres-solver
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
WORKDIR ceres-solver
RUN cmake . && make install
WORKDIR ..
RUN rm -rf ceres-solver

# Install FLANN
RUN apt-get install -y libflann-dev

# Install boost
RUN apt-get update && apt-get install -y libboost-all-dev

# Install pcl
ADD https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.1.tar.gz .
RUN tar xzf pcl-1.8.1.tar.gz
WORKDIR pcl-pcl-1.8.1
RUN mkdir build
WORKDIR build
RUN cmake -D BUILD_keypoints=OFF \
          -D BUILD_ml=OFF \
          -D BUILD_outofcore=OFF \
          -D BUILD_people=OFF \
          -D BUILD_recognition=OFF \
          -D BUILD_registration=OFF \
          -D BUILD_sample_consensus=OFF \
          -D BUILD_segmentation=OFF \
          -D BUILD_simulation=OFF \
          -D BUILD_stereo=OFF \
          -D BUILD_tools=OFF \
          -D BUILD_features=ON \
    ..
RUN make install
WORKDIR ../..
RUN rm -rf pcl*

# Install libproj
RUN apt-get install -y libproj-dev

# Make dynfu build dir
RUN mkdir -p dynfu/build

# Get terra
ADD https://github.com/zdevito/terra/releases/download/release-2016-03-25/terra-Linux-x86_64-332a506.zip .
RUN unzip -qq terra-Linux-x86_64-332a506.zip
RUN mv terra-Linux-x86_64-332a506 terra
RUN ln -s /terra /dynfu/build/terra

# Get Opt
RUN git clone https://github.com/mbrookes1304/Opt.git
WORKDIR Opt/API
RUN git checkout env-variables
RUN make
WORKDIR ../..
RUN ln -s /Opt /dynfu/build/Opt

# Install OpenMesh
ADD http://www.openmesh.org/media/Releases/6.3/OpenMesh-6.3.tar.gz .
RUN tar xzf OpenMesh-6.3.tar.gz
WORKDIR OpenMesh-6.3
RUN mkdir build
WORKDIR build
RUN cmake .. && make install
WORKDIR ../..
RUN rm -rf OpenMesh*
WORKDIR ../..
RUN rm -rf OpenMesh*

# Add source files
ADD CMakeLists.txt /dynfu
ADD cmake /dynfu/cmake
ADD src /dynfu/src
ADD include /dynfu/include

# Build dynfu
WORKDIR dynfu/build
RUN cmake -D CUDA_CUDA_LIBRARY="/usr/local/cuda/lib64/stubs/libcuda.so" ..
RUN make -j`nproc`
WORKDIR ..

# Run dynamicfusion using /data
CMD ./build/bin/app /data
