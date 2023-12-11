# [ICCV2023] HOSNeRF: Dynamic Human-Object-Scene Neural Radiance Fields from a Single Video

  
This is the official repository of **HOSNeRF** [Project page](https://showlab.github.io/HOSNeRF) | [arXiv](https://arxiv.org/abs/2304.12281) | [Video](https://www.youtube.com/watch?v=wS5k5nNkPi4)

[Jia-Wei Liu](https://jia-wei-liu.github.io/), [Yan-Pei Cao](https://yanpei.me),  [Tianyuan Yang](https://scholar.google.com.hk/citations?user=s2q3_A4AAAAJ&hl=zh-CN),  [Zhongcong Xu](https://scholar.google.com/citations?user=-4iADzMAAAAJ&hl=en), [Jussi Keppo](https://www.jussikeppo.com/), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), [Xiaohu Qie](https://scholar.google.com/citations?hl=en&user=mk-F69UAAAAJ&view_op=list_works&sortby=pubdate), [Mike Zheng Shou](https://sites.google.com/view/showlab)

> **TL;DR:** A novel 360° free-viewpoint rendering method that reconstructs neural radiance fields for dynamic human-object-scene from a single monocular in-the-wild video.

<p align="center">
<img src="/assets/HOSNeRF.gif" width="1080px"/>  
<br>
<em>HOSNeRF can render 360° free-viewpoint videos from a single monocular in-the-wild video.</em>
</p>

<p align="center">
<img src="/assets/HOSNeRF.png" width="1080px"/>  
<br>
<em>HOSNeRF Framework.</em>
</p>

## 📢 News

 - [2023.12.11] We release the HOSNeRF codebase!

- [2023.08.16] We release the HOSNeRF dataset!

- [2023.08.12] HOSNeRF got accepted by [**ICCV 2023**](https://iccv2023.thecvf.com/)!

  

- [2023.04.24] We release the arXiv paper!

  

  

## 📝 Preparation

  

### Installation

```
git clone https://github.com/TencentARC/HOSNeRF.git
cd HOSNeRF
pip install -r requirements.txt
```

### Download SMPL model

Download the gender neutral SMPL model from [here](https://smplify.is.tue.mpg.de/), and unpack **mpips_smplify_public_v2.zip**.

Copy the smpl model.

    SMPL_DIR=/path/to/smpl
    MODEL_DIR=$SMPL_DIR/smplify_public/code/models
    cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models of 2nd_State_Conditional_Human-Object and 3rd_Complete_HOSNeRF

Follow [this page](https://github.com/vchoutas/smplx/tree/master/tools) to remove Chumpy objects from the SMPL model.

### HOSNeRF dataset


We release the HOSNeRF dataset on [link](https://drive.google.com/drive/folders/1viuXcihwFpLIjl6TmLyF5VARB7GxEfEv). HOSNeRF dataset consists of 6 real-world dynamic human-object-scene sequences: Backpack, Tennis, Suitcase, Playground, Dance, Lounge. Please run the optical flow estimation method using [RAFT](https://github.com/princeton-vl/RAFT) to get the optical flows of each scene. 

## 🏋️‍️ Experiment

  

### Training

  

**Stage 1: Train the state-conditional background model.**

```bash
$ cd 1st_State-Conditional_Scene
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --ginc configs/state_mipnerf360/Backpack.gin --scene Backpack --logbase 'path to logbase'
```

**Stage 2: Train the state-conditional dynamic human-object model.** 

(Please also change the datadir in configs/default.yaml and core/data/dataset_args.py)

```bash
$ cd 2nd_State_Conditional_Human-Object
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --ginc configs/human-object/Backpack.gin --scene Backpack --logbase 'path to logbase' --cfg configs/human_nerf/wild/monocular/adventure.yaml --seed 777
```

**Stage 3: Train the complete HOSNeRF model using the trained background and human-object checkpoints.** 

(Please also change the datadir in configs/default.yaml and core/data/dataset_args.py)

```bash
$ cd 3rd_Complete_HOSNeRF
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --ginc configs/HOSNeRF/Backpack.gin --scene Backpack --logbase 'path to logbase' --cfg configs/human_nerf/wild/monocular/adventure.yaml --seed 777
```

### Evaluation

We include the test codes in the model's test_step function. It will automatically run the test metrics (`PSNR`, `SSIM`, and `LPIPS`) for test images and all images after training.


### Render 360° free-viewpoint videos

Please change the freeview index in the configs/default.yaml to render the free-viewpoint videos of that timestep. It will automatically render free-viewpoint videos (the model's test_step function) after training.


### Render canonical human-object videos

It will automatically render the canonical human-object videos (the model's test_step function) after training.


### Resume training

To resume training or resume testing after training, please add --resume_training True for each training script.


### HOSNeRF checkpoints

We release the 6 HOSNeRF checkpoints on [link](https://drive.google.com/drive/folders/15I7z7qjBL6rQ3z91_rzhR284L1vfxxOX?usp=drive_link) for reference.


## 🎓 Citation

If you find our work helps, please cite our paper.

```bibtex
@inproceedings{liu2023hosnerf,
  title={Hosnerf: Dynamic human-object-scene neural radiance fields from a single video},
  author={Liu, Jia-Wei and Cao, Yan-Pei and Yang, Tianyuan and Xu, Zhongcong and Keppo, Jussi and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18483--18494},
  year={2023}
}
```

## ✉️ Contact
This repo is maintained by [Jiawei Liu](https://jia-wei-liu.github.io/). Questions and discussions are welcome via jiawei.liu@u.nus.edu.

## 🙏 Acknowledgements
This codebase is based on [HumanNeRF](https://github.com/chungyiweng/humannerf) and [NeRF-Factory](https://github.com/kakaobrain/nerf-factory). The preprocessing code is based on [NeuMan](https://github.com/apple/ml-neuman). Thanks for open-sourcing!

## LICENSE
Copyright (c) 2023 Show Lab, National University of Singapore. All Rights Reserved. Licensed under the Apache License, Version 2.0 (see [LICENSE](https://github.com/TencentARC/HOSNeRF/blob/main/LICENSE) for details)