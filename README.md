# HFFST: A Hierarchical Feature Fusion Algorithm for Spatial Gene Expression Prediction using Histopathology Images
We propose HFFST, a hierarchical feature fusion algorithm for predicting spatial gene expression from H\&E-stained pathology images. Leveraging multi-level feature extraction and fusion from whole-slide images, our method employs a coarse-to-fine regression framework to predict spatial transcriptomic profiles. To validate the algorithm's performance, we conducted cross-validation on three public datasets and performed external validation using high-resolution Visium data from 10X Genomics. HFFST has shown promise in predicting spatial gene expression and identifying spatial regions, demonstrating certain advantages over state-of-the-art methods.


## Structure
  "'
  markdown
  ![The overall structure](https://imgur.com/a/pMnO06f)
  '"


## Code Reference & Citation  
This implementation is based on the publicly available code from [TRIPLEX](https://github.com/NEXGEM/TRIPLEX).  
The original work is described in the paper:  

@inproceedings{chung2024accurate,
  title={Accurate Spatial Gene Expression Prediction by integrating Multi-resolution features},
  author={Chung, Youngmin and Ha, Ji Hun and Im, Kyeong Chan and Lee, Joo Sang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11591--11600},
  year={2024}
}

We appreciate the authors for sharing their code and research.  
