# Adversarial_Patch_Attack
Pytorch implementation of Adversarial Patch on ImageNet (arXiv: https://arxiv.org/abs/1712.09665)

## Experiment Result
We selected Pytorch pretrained model ResNet50 as our victim model.  
We generate the patch on 2000 randomly selected pictures with 50 epochs and different size of noise.  
After generated, the patch is tested on 1000 rondomly selected pictures.  
The successful attack rate of our best patch is in the chart below.  

|noise percentage | 0.5 | 0.6 |
|:----: | :----: |:----:|
|patch size | (50, 50) | (54, 54) | 
|successful rate | 98.48% | 99.61% |

### Adversarial Patch
One of our found best patch is shown below.  


![best_patch](https://github.com/zhaojb17/Adversarial_Patch_Attack/blob/master/experiment_statistics/5%25noise/pictures/best_patch.png)

## Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer  
    [Adversarial Patch. arXiv:1712.09665](https://arxiv.org/abs/1712.09665)
