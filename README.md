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
|#RegDB      | ~ 91.6%  | ~ 84.1% |
|#SYSU-MM01  | ~ 70.6%  | ~ 66.9% |


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
Our code is based on [mangye16](https://github.com/mangye16/Cross-Modal-Re-ID-baseline). 

Contact: zhangyk@stu.xmu.edu.cn
