Dynfu

[Dynfu overview and report](https://spina.me/assets/files/dynamic-fusion.pdf)

============
Dependencies:
* CUDA-enabled CPU (Kepler or newer) with CUDA 7.5 or higher, required by Opt
* OpenCV 2.4.9 with new Viz module (only opencv_core, opencv_highgui, opencv_imgproc, opencv_viz modules required); make sure that WITH_VTK flag is enabled in CMake during OpenCV configuration

Implicit dependency (needed by opencv_viz):
* VTK 5.8.0 or higher (apt-get install on linux, for Windows please download and compile from www.vtk.org)

## Future additions
* Surface fusion using PSDF for non-rigid surfaces
* Exending the warpfield by adding in new deformation nodes
* Regularisation for the energy function

## References
The CPU warpfield solver is based on [Ceres](https://github.com/ceres-solver/ceres-solver).
```
@misc{ceres-solver,
  author = "Sameer Agarwal and Keir Mierle and Others",
  title = "Ceres Solver",
  howpublished = "\url{http://ceres-solver.org}",
}
```
The GPU warpfield solver is based on [Opt](http://optlang.org).
```
@article{devito2016opt,
  title={Opt: A Domain Specific Language for Non-linear Least Squares Optimization in Graphics and Imaging},
  author={DeVito, Zachary and Mara, Michael and Zoll{\"o}fer, Michael and Bernstein, Gilbert and Theobalt, Christian and Hanrahan, Pat and Fisher, Matthew and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:1604.06525},
  year={2016}
}
```
The KD-tree is created using [nanoflann](https://github.com/jlblancoc/nanoflann).
```
@misc{blanco2014nanoflann,
  title        = {nanoflann: a {C}++ header-only fork of {FLANN}, a library for Nearest Neighbor ({NN}) wih KD-trees},
  author       = {Blanco, Jose Luis and Rai, Pranjal Kumar},
  howpublished = {\url{https://github.com/jlblancoc/nanoflann}},
  year         = {2014}
}
```
