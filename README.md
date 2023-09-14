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

<div id="default-editor-page__placeholder" class="panel">
    <link rel="stylesheet" href="https://template-1253409072.cos.ap-guangzhou.myqcloud.com/vip-tpl/assets/css/plugins.css"/>
    <link rel="stylesheet" href="https://template-1253409072.cos.ap-guangzhou.myqcloud.com/vip-tpl/assets/css/style.css"/>
    <section class="wrapper bg-gradient-primary">
      <img src="assets/animation_occluded_sample1.gif" width="200" height="200" />
    </section>
    <section class="wrapper bg-gradient-primary">
    </section>
  </div>
  <img src="https://static.htmlpage.cn/editor/images/assets/bg4.jpg" class="c3070"/>
  <div class="htmlpage-row">
    <div class="htmlpage-cell">
    </div>
    <div class="htmlpage-cell">
    </div>
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

* {
  box-sizing: border-box;
}
body {
  margin: 0;
}
*{
  box-sizing:border-box;
}
body{
  margin-top:0px;
  margin-right:0px;
  margin-bottom:0px;
  margin-left:0px;
  font-family:-apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Roboto, Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft Yahei", "Microsoft Jhenghei", sans-serif;
}
.panel{
  color:rgb(51, 51, 51);
  font-weight:100;
}
.logo path{
  pointer-events:none;
  fill:none;
  stroke-linecap:round;
  stroke-width:7;
  stroke:rgb(255, 255, 255);
}
.wechat-group img{
  max-width:220px;
  height:auto;
  border-top-left-radius:8px;
  border-top-right-radius:8px;
  border-bottom-right-radius:8px;
  border-bottom-left-radius:8px;
  margin-top:0px;
  margin-right:auto;
  margin-bottom:0px;
  margin-left:auto;
  width:100%;
}
.welcome-img img{
  width:100%;
}
.bg-gradient-primary{
  border-top-left-radius:8px;
  border-top-right-radius:8px;
  border-bottom-right-radius:8px;
  border-bottom-left-radius:8px;
}
.fdb-block img{
  border-top-left-radius:4px;
  border-top-right-radius:4px;
  border-bottom-right-radius:4px;
  border-bottom-left-radius:4px;
}
.fdb-block .fdb-touch{
  border-top-width:5px;
  border-top-style:solid;
  border-top-color:rgb(82, 139, 255);
}
.fdb-block .fdb-box{
  background-image:initial;
  background-position-x:initial;
  background-position-y:initial;
  background-size:initial;
  background-repeat-x:initial;
  background-repeat-y:initial;
  background-attachment:initial;
  background-origin:initial;
  background-clip:initial;
  background-color:rgb(255, 255, 255);
  color:rgb(68, 68, 68);
  padding-top:60px;
  padding-right:40px;
  padding-bottom:60px;
  padding-left:40px;
  border-top-left-radius:4px;
  border-top-right-radius:4px;
  border-bottom-right-radius:4px;
  border-bottom-left-radius:4px;
  box-shadow:rgba(0, 0, 0, 0.12) 0px 1px 3px, rgba(0, 0, 0, 0.24) 0px 1px 2px;
  overflow-x:hidden;
  overflow-y:hidden;
}
a{
  color:rgb(0, 123, 255);
  text-decoration-line:none;
  text-decoration-thickness:initial;
  text-decoration-style:initial;
  text-decoration-color:initial;
  background-color:transparent;
}
a:hover{
  color:rgb(0, 86, 179);
  text-decoration-line:underline;
  text-decoration-thickness:initial;
  text-decoration-style:initial;
  text-decoration-color:initial;
}
.fdb-block .text-h1, .fdb-block h1{
  font-size:1.75rem;
  margin-bottom:0.5em;
  margin-top:0.3em;
  font-weight:400;
}
.c3070{
  color:black;
}
.htmlpage-row{
  display:table;
  padding-top:10px;
  padding-right:10px;
  padding-bottom:10px;
  padding-left:10px;
  width:100%;
}
.htmlpage-cell{
  width:8%;
  display:table-cell;
  height:75px;
}
@media (max-width: 768px){
  .htmlpage-cell{
    width:100%;
    display:block;
  }
}

