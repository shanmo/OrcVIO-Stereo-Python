# about 

- this repo implements the stereo version of OrcVIO 

# Requerements

* Python 3.6+
* numpy
* scipy
* cv2
* [pangolin](https://github.com/uoip/pangolin) (optional, for trajectory/poses visualization)
* [sophuspy](https://pypi.org/project/sophuspy/), check [this](https://github.com/pybind/pybind11/issues/1628#issuecomment-697346676) for how to install 

# how to run  

- change the dataset path in vio.py and run `python vio.py`  
- to speed up, use `load_features_flag` to load saved features 

# results

- [demo video for MH_01_easy](https://youtu.be/eNaVp7B5ecQ)  
![](imgs/euroc_mh_01_easy.jpg)

# references 

- https://github.com/uoip/stereo_msckf
