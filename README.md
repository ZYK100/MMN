# Towards-a-Unified-Middle-Modality-Learning-for-Visible-Infrared-Person-Re-Identification

[Paper](https://dl.acm.org/doi/10.1145/3474085.3475250)

This repository is Pytorch code for our proposed MMN method for Cross-Modality Person Re-Identification. 


### Training.
  Train a model by
  ```bash
python train.py --dataset sysu
```

  - `--dataset`: which dataset "sysu" or "regdb".

### Result.

The results may have some fluctuation, and might be better by finetuning the hyper-parameters.


|Datasets    | Rank@1   | mAP     |
| --------   | -----    |  -----  |
|#RegDB[1]      | ~ 91.6%  | ~ 84.1% |
|#SYSU-MM01[2]  | ~ 70.6%  | ~ 66.9% |


### Citation

Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{zhang2021towards,
  title={Towards a Unified Middle Modality Learning for Visible-Infrared Person Re-Identification},
  author={Zhang, Yukang and Yan, Yan and Lu, Yang and Wang, Hanzi},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={788--796},
  year={2021}
}
```

Our code is based on [mangye16](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [3, 4]. 

###  References.


[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[3] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C., Hoi. 	Deep learning for person re-identification: A survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.

[4] M. Ye, X. Lan, Z. Wang, and P. C. Yuen. Bi-directional Center-Constrained Top-Ranking for Visible Thermal Person Re-Identification. IEEE Transactions on Information Forensics and Security (TIFS), 2019.


If you have any question, please feel free to contact us. zhangyk@stu.xmu.edu.cn.
