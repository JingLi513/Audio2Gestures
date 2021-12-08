# Audio2Motion
Audio2Motion Official implementation for Audio2Motion: Generating Diverse Gestures from Speech with Conditional Variational Autoencoders.

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
