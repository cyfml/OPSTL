<div align="center">
<p align="center">OPSTL：Unveiling the Hidden Realm: Self-supervised Skeleton-based Action Recognition in Occluded Environments
<br>

<div align="center">
  Yifei&nbsp;Chen</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Kunyu-Peng" target="_blank">Kunyu&nbsp;Peng</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Alina-Roitberg-2" target="_blank">Alina&nbsp;Roitberg</a> <b>&middot;</b>
  David&nbsp;Schneide</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Jiaming-Zhang-10" target="_blank">Jiaming&nbsp;Zhang</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Junwei-Zheng-4" target="_blank">Junwei&nbsp;Zheng</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Ruiping-Liu-7" target="_blank">Ruiping&nbsp;Liu</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Yufan-Chen-27" target="_blank">Yufan&nbsp;Chen</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Kailun-Yang" target="_blank">Kailun&nbsp;Yang</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Rainer-Stiefelhagen" target="_blank">Rainer&nbsp;Stiefelhagen</a>
 <br>

  <a href="https://github.com/cyfml/OPSTL" target="_blank">Paper</a>

# 

</div>

<p align="center">:hammer_and_wrench: :construction_worker: :rocket:</p>
<p align="center">:fire: We will release code in the future. :fire:</p>
<div style="text-align: center;">
  <img src="assets/animation_occluded_sample1.gif" width="300" height="300" />
                                             
  <img src="assets/animation_imputed_sample1.gif" width="300" height="300" />
</div>
<div style="text-align: center;">
  <img src="assets/animation_occluded_sample2.gif" width="300" height="300" />
                                             
  <img src="assets/animation_imputed_sample2.gif" width="300" height="300" />
</div>

<!-- <div align=left><img src="assets/animation_occluded_sample1.gif" width="200" height="200" />
</div><div align=left><img src="assets/animation_occluded_sample1.gif" width="200" height="200" /></div> -->

### Update

- 2023.09.14 Init repository.



### TODO List

- [ ] Code release. 

### Abstract

Abstract— In order to integrate action recognition methods into autonomous robotic systems, it is crucial to consider adverse situations involving target occlusions. Such a scenario, despite its practical relevance, is rarely addressed in existing self-supervised skeleton-based action recognition methods. To empower robots with the capacity to address occlusion, we propose a simple and effective method. We first pre-train using occluded skeleton sequences, then use k-means clustering (KMeans) on sequence embeddings to group semantically similar samples. Next, we employ K-nearest-neighbor (KNN) to fill in missing
skeleton data based on the closest sample neighbors. Imputing incomplete skeleton sequences to create relatively complete sequences as input provides significant benefits to existing skeleton-based self-supervised models. Meanwhile, building on the state-of-the-art Partial Spatio-Temporal Learning (PSTL), we introduce an Occluded Partial Spatio-Temporal Learning (OPSTL) framework. This enhancement utilizes an Adaptive Spatial Masking (ASM) for a better use of high-quality, intact skeletons. The effectiveness of our imputation methods is verified on the challenging occluded versions of the NTURGB+D 60 and NTURGB+D 120.

### Method

<p align="center">
    (Overview)
</p>
<p align="center">
    <div align=center><img src="assets/Figtwo.jpg" width="850" height="330" /></div>
<br><br>

### Contact

Feel free to contact me if you have additional questions or have interests in semantic segmentation based on light field camera. Please drop me an email at cyf236510120@gmail.com
