# Human Pose Estimation 3D in Unity

## Introduction

This repository uses [MobileHumanPose](https://github.com/SangbumChoi/MobileHumanPose) network to get the pose of a person in 3D space. The proposed approach combines the TFLite backend and the Unity engine. As a result, it is possible to get 7-8 FPS from the 640x480 video camera of the top Snapdragon 8+ Gen 1 processor.

TFLite uses NNAPI and runs on the CPU, however it is possible to run on the GPU, which is not yet supported due to the lack of some operators. Important: the neural network `.tflite` file is taken from the official [MobileHumanPose](https://github.com/SangbumChoi/MobileHumanPose) repository, respectively, and the executable graph is taken from it without changes.
 
## Dependencies

### OpenCV for Unity

To launch this repository, a paid asset [OpenCV for Unity](https://assetstore.unity.com/packages/tools/integration/opencv-for-unity-21088) from the Unity store is used. This package also has a [free version](https://assetstore.unity.com/packages/tools/integration/opencv-plus-unity-85928), but it is not compatible with the code in this repository.

This asset performs the task of detecting a person in the frame, as well as tracking the found bounding box. The repository also uses the interface elements of this package.

### TensorFlow Lite libraries

To support TFLite on Unity, the code from the following [repository](https://github.com/asus4/tf-lite-unity-sample) is used. To quickly connect the required module to the project, use OpenUPM:

* "com.github.asus4.tflite": "2.12.0"
* "com.github.asus4.tflite.common": "2.12.0"
