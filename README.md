Dynfu
============
Dependencies:
* CUDA-enabled CPU (Kepler or newer) with CUDA 5.0 or higher
* OpenCV 2.4.9 with new Viz module (only opencv_core, opencv_highgui, opencv_imgproc, opencv_viz modules required). Make sure that WITH_VTK flag is enabled in CMake during OpenCV configuration.

Implicit dependency (needed by opencv_viz):
* VTK 5.8.0 or higher. (apt-get install on linux, for windows please download and compile from www.vtk.org)

## Future additions
* Surface fusion using PSDF for non-rigid surfaces
* Exending the warpfield by adding in new deformation nodes
* Regularisation for the energy function
* GPU solver for the warpfield
* Visualiser for the canonical frame warped to live

## References
The warpfield solver is based on [Ceres](https://github.com/ceres-solver/ceres-solver)
```
@misc{ceres-solver,
  author = "Sameer Agarwal and Keir Mierle and Others",
  title = "Ceres Solver",
  howpublished = "\url{http://ceres-solver.org}",
}
```
The KD-tree is created using [nanoflann](https://github.com/jlblancoc/nanoflann)
```
@misc{blanco2014nanoflann,
  title        = {nanoflann: a {C}++ header-only fork of {FLANN}, a library for Nearest Neighbor ({NN}) wih KD-trees},
  author       = {Blanco, Jose Luis and Rai, Pranjal Kumar},
  howpublished = {\url{https://github.com/jlblancoc/nanoflann}},
  year         = {2014}
}
```
