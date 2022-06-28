# Audio2Gestures
Official implementation for Audio2Gestures: Generating Diverse Gestures from Speech Audio with Conditional Variational Autoencoders， ICCV

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
Run script ``python .\smplx2fbx.py --smplx .\for_smplx_retargeting.h5  --key LclRotation --fps 30  --synthesized .\input.h5 --fbx output.fbx`` 

## Reference
If you find our code useful for your research, please cite our paper.
> @inproceedings{li2021audio2gestures,  
>  title={Audio2Gestures: Generating Diverse Gestures from Speech Audio with Conditional Variational Autoencoders},  
>  author={Li, Jing and Kang, Di and Pei, Wenjie and Zhe, Xuefei and Zhang, Ying and He, Zhenyu and Bao, Linchao},  
>  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},  
>  pages={11293--11302},  
>  year={2021}  
> }
