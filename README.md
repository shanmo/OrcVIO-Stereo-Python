# about 

- this repo implements the stereo version of OrcVIO used in [OrcVIO: Object residual constrained Visual-Inertial Odometry](http://me-llamo-sean.cf/orcvio_githubpage/), the related papers are: 

- [IROS 2020](https://arxiv.org/abs/2007.15107)
- [Journal version TBD]()

If you find this repo useful, kindly cite our publications 

```
@inproceedings{orcvio,
	  title = {OrcVIO: Object residual constrained Visual-Inertial Odometry},
          author={M. {Shan} and Q. {Feng} and N. {Atanasov}},
          year = {2020},
          booktitle={IEEE Intl. Conf. on Intelligent Robots and Systems (IROS).},
          url = {http://erl.ucsd.edu/pages/orcvio.html},
          pdf = {https://arxiv.org/abs/2007.15107}
}
```

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
