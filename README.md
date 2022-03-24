# Audio2Gestures
Official implementation for Audio2Gestures: Generating Diverse Gestures from Speech Audio with Conditional Variational Autoencoders

## Dependencies
- [pytorch](https://pytorch.org/)
- [numpy](https://numpy.org/)
- [librosa](https://librosa.org/)
- [fbx](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-0)
- [smplx 1.0](https://github.com/vchoutas/smplx)

## Data Processing
1. Download fbx format data from [Trinity speech gesture](https://trinityspeechgesture.scss.tcd.ie/).
2. Retarget the data into SMPLX model using maya.
3. convert the fbx data to hdf5 format using script ``fbx2hdf.py``


## Training and testing
Run script ``bash start.sh``

## Visualizing
Run script ``smplx2hdf.py``

## Reference
If you find our code useful for your research, please cite our paper.
> @article{2021Audio2Gestures,
>   title={Audio2Gestures: Generating Diverse Gestures from Speech Audio with Conditional Variational Autoencoders},
>   author={ Li, J.  and  Kang, D.  and  Pei, W.  and  Zhe, X.  and  Bao, L. },
>   year={2021},
> }
