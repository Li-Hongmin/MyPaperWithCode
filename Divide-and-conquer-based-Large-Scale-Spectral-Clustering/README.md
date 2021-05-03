# Divide-and-conquer based Large-Scale Spectral Clustering

You can find our paper in [ArXiv.org](https://arxiv.org/abs/2104.15042) Or [ResearchGate](http://dx.doi.org/10.13140/RG.2.2.15207.37281).

## Datasets
Five real world datasets and five synthetic datasets:

- PenDigits 
- USPS 
- Letters 
- MNIST 
- Covertype 
- Three Spirals-60K (TS-60K)
- Two Moons-1M (TM-1M)
- Three Circles-6M (TC-6M)
- Circles and Gaussian-10M (CG-10M) 
- Flower-20M (FL-20M) 

## Algorithm 


- [Divide-and-conquer based Large-scale Spectral Clustering](DnC_SC.m)

![figs](figs/overview.jpg "An overview of our method.")


- [Divide-and-conquer based Landmark Selection](DnC_landmark)
    ![figs](figs/dnc_landmark_selection.jpg "An illustration of divide-and-conquer based landmark selection.")
    <!-- - [light-k-means](figs) -->
- Approximate K-nearest landmarks method (line 35~62 of [DnC_SC](DnC_SC.m))
    ![figs](figs/aknn.jpg "An approximate K-nearest landmarks method.")

## Code 

See our [demo](demo.m).

# Reference
If you find this code useful for your research, please cite
```
@misc{li2021divideandconquer,
    title={Divide-and-conquer based Large-Scale Spectral Clustering},
    author={Hongmin Li and Xiucai Ye and Akira Imakura and Tetsuya Sakurai},
    year={2021},
    eprint={2104.15042},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```