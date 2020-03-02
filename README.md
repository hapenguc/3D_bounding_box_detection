# Using CenterNet and Deep3DBox for 3D bounding box detection



## Main results

### Object Detection on kitty validation

| Backbone     |  AP /E | AP /M             |  AP / H |
|--------------|-----------|--------------|-----------------------|
|ResNet-18     | 	89.22%       | 	78.71%             |           69.8%	|


### 3D bounding box detection on KITTI validation

|Backbone|AP-E|AP-M|AP-H|
|--------|---|----|----|
|ResNet-18  |15.14% | 13.34| 10.89%  |



## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use CenterNet

We support demo for image/ image folder


For 3D object detection on images/ video, run:

~~~
python demo.py ddd--demo /path/to/image/or/folder/or/video --load_model path/to/model  
~~~

You can add `--debug 2` to visualize the heatmap outputs.

To use this CenterNet in your own project, you can 

~~~
import sys
CENTERNET_PATH = /path/to/CenterNet/src/lib/
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = /path/to/model
TASK = 'ddd'
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

img = image/or/path/to/your/image/
ret = detector.run(img)['results']
~~~
`ret` will be a python dict: `{category_id : [[x1, y1, x2, y2, score], ...], }`

## License

CenterNet itself is released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from [human-pose-estimation.pytorch](https://github.com/Microsoft/human-pose-estimation.pytorch) (image transform, resnet), [CornerNet](https://github.com/princeton-vl/CornerNet) (hourglassnet, loss functions), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2)(deformable convolutions), [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)(Pascal VOC evaluation) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).


