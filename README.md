# fast-clip-research

# A short conclusion on the speed test

## **********TLDR**********

The test is done on g4 Sagemaker Instance with `device = "cuda"`

**Inference**

1. `"load:cpu-ViT-L/14"` is our current choice for clip model. Note that even if we pass the device `"cuda"` to the serve, the model is still firstly loaded from `"cpu"` and then moved to `"cuda"` . This is to guarantee that all the operation are implemented in `torch.float32` precision. 
2. `"onnx/ViT-L/14"` onnxruntime-gpu with CUDAExecutionProvider is almost the free lunch. It can produce exactly the same results as 1. It can only require some extra storage space to store the onnx models, which is 1GB around.
3. `"load:cuda-ViT-L/14"` this methods loads the clip model directly into `"cuda"`, which makes all the computations are done in `torch.float16.` This can increase the inference speed but introduce a slightly difference.

The inference speed and `add_document()` time cost as well as searching performance are shown in the table below:
| Model Name | Inference Time | Image Indexing Time (CBS = 50) | Text Indexing Time (CBS = 100) | Text2Image Score | Image2Text Score | device|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| load:cpu-ViT-L/14 | 66ms | 84ms | 10.5ms | 95 | 69 |cuda|
| load:gpu-ViT-L/14 | 22ms | 34ms | 9.2ms | 95 | 68 |cuda|
| onnx/ViT-L/14 | 55ms | 67ms | 9.5ms | 95 | 69 |cuda|
| onnx16/ViT-L/14 |  19ms |   28ms   |       |    |    |cuda|

**Recommendation**s: encourage users to  `"onnx"` version to index the documents. `"load:cuda"` can be added but we should tell the users they may get different results.

**Image Preprocessing**

1. `clip.preprocess` the default clip image preprocessing takes the `PIL.Image` as input and process using the `PIL` functions. 
2. One option is to replace all the `PIL.Processing` by OpenCV package `cv2` . It reads the image as `ndarray` and process it. This faster but unfortunately, results are different.

The preprocessing speed and there search performance are shown as below. Note that the preprocessing speed varies when different input images are passed. 

| Preprocessing Method | TIME (ms) (PNG File with size = (2162, 762)) | TIME (ms) (JPG File with size = (640, 425)) | Text2Image Score | Image2Text Score |
|:---:|:---:|:---:|---|---|
| original_clip | 27.4 ms ± 94.8 µs | 4.39 ms ± 15 µs | 97.5 | 91 |
| opencv | 672 µs ± 143 µs | 652 µs ± 70.4 µs | 90 | 88 |

**Recommendations:** ask the customer the provide the image with a small size and RBG channels (.jpg). This will reduce the preprocessing time in the indexing. I wouldn’t think we should add the opencv preprocessing into our model as the differences are relative large and noticeable. 

## Intro

We focus the inference speed of clip model here. 

When we aim to index a document (whether a image or text), the document is passed to the clip model to convert it to tensors by calling the function `vectorise()` . In our settings, every time we call `vectorise()` , only one document is passed into the neural network model. For example, if we want to index 1,000 documents, the `vectorise()` will be called 1,000 times. 

The reason that we emphasise this point is to specify the situation that we want to accelerate:  

**single batch with one example.** 

This special situation disables a lot of  widely used batching based accelerating methods. We should always put this special in our mind when we test the accelerating methods.

While `clip_models` can take both images and text as inputs and vectorise them into tensors, we mainly focus on the image inputs as they are time consuming. For comparison, an image may take about 80ms to vectorise while a text sentence may take only 10ms. 

## Image Preprocessing

When an image is passed to the clip_model, there are two step, 

- 1) image preprocessing,
    
    2) inference 
    

The image preprocessing step will convert the input to a 3-channel (RGB) normalised image (tensor) with size 224X224X3, however the input size or channel. The code for image preprocessing is:

```python
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor

def _convert_image_to_rgb(image):
    return image.convert("RGB")

torchvision_transform = Compose([
		    #torchvision transform normally takes PIL Image as input
        Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
```

The required input type is `PIL.Image.` 

**RGB Conversion**

Note that the time consuming steps `Resize` and `CenterCrop` are implemented before the RGB conversion. This means if an image has 4 channels (e.g., “.png” image with RGBa), these time consuming steps are implemented on a 4 dimensional tensor, which takes longer time than on a 3 dimensional tensor. Following this, one simple way to speeding up the preprocessing step is to swap the order of RGB conversion and those resizing steps, which we have:

```python
rgb_transform= Compose([
# We convert the image to rgb first before resize, centercrop, etc._convert_image_to_rgb,
    Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
```

Note that the results will be different for non-RGB images.

**OpenCV Conversion**

As we aforementioned, the preprocessing is conducted under `PIL.Image` format. Another widely used image preprocessing package is `OpenCV` . It is not hard to reimplement the whole process in `OpenCV` by the following code with the `augmennt` (package)[https://github.com/wanliAlex/augmennt]. 

```python
cv_transform= at.Compose([
    at.Resize(224, interpolation= "BICUBIC"),
    at.CenterCrop(224),
    _convert_bgr_to_rgb,
    at.ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
```

This is super fast, especially when the input size is large. However, it will introduce a large difference in the result.

## Inference





# Large Scale Test 

| Model Name | Image Indexing Time (CBS = 50) | Text Indexing Time (CBS = 100) | Text2Image Score | Image2Text Score | Image2Image |
|---|---|---|---|---|---|
| load-cpu: ViT-L/14 | 84ms | 10.5ms | 95 | 69 | [link]() |
| load-gpu: ViT-L/14 | 34ms | 9.2ms | 95 | 68 | [link]() |
| onnx-float16-opencv | xx | xx | xx | xx | [link]() |
| onnx32 | 67ms | 9.5ms | 95 | 69 | [link]() |

For each of these tests, each document consists of a single field (text or image).

The images were locally hosted on a Python image server.

# EC2 Instance
### Add_Documents() Time

**Note**:
- CBS == client_batch_size 

| Model Name | Image Indexing Time (CBS = 100) | Text Indexing Time (CBS = 100) | Image Indexing Time (CBS = 50) | Text Indexing Time (CBS = 50) | Image Indexing Time (CBS = 10) | Text Indexing Time (CBS = 10) | Image Indexing Time (CBS = 1) | Text Indexing Time (CBS = 1) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Vit-B/32 * | 18 | 7 | 19 | 8 | 26 | 14 | 70 | 65 |
| fast/Vit-B/32 ** | 17 | 6 | 36 | 8 | 44 | 14 | 80 | 80 |
| Vit-L/14 | 74 | 9 | 74 | 11 | 80 | 15 | 129 | 65 |
| fast/Vit-L/14 | 58 | 9 | 410 | 10 | 420 | 28 | 500 | 139 |
| openclip/Vit-L/14 | 76 | 11.8 | 78 | 13 | 89 | 22 | 220 | 14 |
| opencv/Vit-L-14/cuda | 73 | 9 | 77 | 11 | 88 | 15 | 218 | 65 |
| opencv/Vit-L-14/trt todo | 73 | 9 | 77 | 11 | 88 | 15 | 218 | 65 |
| onnx/ViT-L/14 | 64 | 9 | 60 | 10 | 71 | 28 | 226 | 139 |

For __onnx/ViT-L/14__, there is a converging process in the processing speed. The indexing time starts from 150m/per doc and converges to 64ms/per doc after 40 batches.

### Inference Speed

| Models | Time cost | Difference (mean difference normalized by dimension) | Comments |
|---|---|---|---|
| load:cpu - ViT-L/14 | 66.2 ms ± 309 µs | N/A | The very original clip version, output.dtype: torch.float32 |
| load:cuda - ViT-L/14 | 18.7 ms ± 245 µs | 3.6e-3 | We load the model from cuda, output.dtype: torch.float16 |
| load:cuda - mix precision - ViT-L/14 | 27.3 ms ± 335 µs | 3.6e-3 | we use `torch.autocast(device_type='cuda', dtype=torch.float16)` to do inference |
| open-clip/ViT-L/14 | 66.9 ms ± 435 µs | N/A | This is a more reasonable speed on pytorch |
| cuda:onnx/ViT-L/14 | 55.7 ms ± 166 µs | 9e-6 | Using clip_onnx package |
| tensorrt:onnx/ViT-L/14 | 47.7 ms ± 639 µs | 9e-6 | The environment is really unstable，it has very strict requirements on onnxruntime, cuda, tensorrt version |
| TorchDynam | 21 ms ± 234 µs | N/A | Basicly this is just another version of onnx or tensorrt, so it is not helping, [link](https://github.com/pytorch/torchdynamo) |
| kernlai |  |  | It requires python>3.9 and gpu capability > 8, g5 instancem maybe, [link](https://github.com/ELS-RD/kernl) |



### Preprocessing Speed

| TRANSFORMS | TIME (ms) (PNG File with size = (2162, 762)) | TIME (ms) (JPG File with size = (640, 425)) | Comments |
|---|---|---|---|
| original_clip | 27.4 ms ± 94.8 µs | 4.39 ms ± 15 µs |  |
| our_clip_implementation | 27.4 ms ± 49.8 µs | 4.4 ms ± 16.8 µs |  |
| opencv_based | 4.8 ms ± 194 µs | 1.08 ms ± 3.02 µs |  |
| script_based | 11.8 ms ± 51.2 µs | 2.26 ms ± 21.1 µs |  |
| rgb_conversion | 18.4 ms ± 28.4 µs | 4.47 ms ± 13 µs |  |
| grey_conversion | 12.7 ms ± 15.5 µs | 3 ms ± 60.1 µs |  |
| read_from_cv | 672 µs ± 143 µs | 652 µs ± 70.4 µs |  |

### Performance
| Model Name | Text-to-image score (single-label) | Text-to-image score (double-label) | Text-to-image (trible-label) | Image-to-text score | Image-to-Image score |
|---|---|---|---|---|---|
| Vit-B/32 | 92.5 | 78.75 | 46.7 | 91 | good |
| Vit-L/14 | 97.5 | 82.5 | 52.3 | 91 | good |
| fast/Vit-B/32 | 97.5 | 72.5 | 48 | 88 | good |
| fast/Vit-L/14 | 90 | 81.25 | 52.3 | 88 | good |
| openclip/Vit-L/14 | 97.5 | 82.5 | 52.3 | 91 | good |
| opencv/Vit-L-14 | 90 | 81.25 | 52.3 | 88 | good |
| onnx/ViT-L/14 | 97.5 | 82.5 | 52.3 | 91 | good |


# Inference Speed breakdown

## Time Break Down for function `vectorise( )`

## For the pytorch “Vit-L/14” ,

### if we load the model from “cpu”, which is float32

INFO:marqo.s2_inference.s2_inference:The client gives 1 documents to vectorise

INFO:marqo.s2_inference.clip_utils:It takes about 0.005s to load all images. The average time for each image is 0.005s

INFO:marqo.s2_inference.clip_utils:It takes about 0.005s to preprocess all images. The average time for each image is 0.005s

INFO:marqo.s2_inference.clip_utils:It take about 0.011s to encode all images. The average time for each image is 0.011s

INFO:marqo.s2_inference.clip_utils:It takes 0.049s to convert the output with `float32` to ndarray from cuda

INFO:marqo.s2_inference.s2_inference:It take about 0.071s to vectorise all documents. The average time for each document is 0.071s

### if we load the model from “cuda”, which is float16

INFO:marqo.s2_inference.s2_inference:The client gives 1 documents to vectorise

INFO:marqo.s2_inference.clip_utils:It takes about 0.005s to load all images. The average time for each image is 0.005s

INFO:marqo.s2_inference.clip_utils:It takes about 0.005s to preprocess all images. The average time for each image is 0.005s

INFO:marqo.s2_inference.clip_utils:It take about 0.012s to encode all images. The average time for each image is 0.012s

INFO:marqo.s2_inference.clip_utils:It takes 0.004s to convert the output with `float16` to ndarray from cuda

INFO:marqo.s2_inference.s2_inference:It take about 0.026s to vectorise all documents. The average time for each document is 0.026s

np.abs(np a - np b).sum().  0.13

### if we load the model from “cpu” but cast it to float16

INFO:marqo.s2_inference.s2_inference:The client gives 1 documents to vectorise

INFO:marqo.s2_inference.clip_utils:It takes about 0.005s to load all images. The average time for each image is 0.005s

INFO:marqo.s2_inference.clip_utils:It takes about 0.005s to preprocess all images. The average time for each image is 0.005s

INFO:marqo.s2_inference.clip_utils:It take about 0.011s to encode all images. The average time for each image is 0.011s

INFO:marqo.s2_inference.clip_utils:It takes 0.051s to convert the output with `float16` to ndarray from `cuda`

INFO:marqo.s2_inference.s2_inference:It take about 0.072s to vectorise all documents. The average time for each document is 0.072s






# M1 Mac

## Add documents part

We test the time of adding a document into index under different batch sizes.

### Timing Test (On M1 Mac Device = "CPU")

__CBS___ = Client_Batch_Size

| Model Name | Image Indexing Time (CBS = 50) | Text Indexing Time (CBS = 50) | Image Indexing Time (CBS = 10) | Text Indexing Time (CBS = 10) | Image Indexing Time (CBS = 1) | Text Indexing Time (CBS = 1) |
|---|--------------------------------|---|---|---|---|---|
| Vit-B/32 * | 64                             | 41 | 66 | 64 | 117 | 171 |
| Vit-L/14 | 335                            | 55 | 345 | 61 | 672 | 128 |
| fast/Vit-B/32 ** | __36__                         | 22 | 44 | 27 | 80 | 80 |
| fast/Vit-L/14  | 410                            | 41 | 420 | 48 | 500 | 95 |
| openclip/Vit-L/14 | __295__                        | 52 | 306 | 63 | 360 | 105 |
| opencv/Vit-L-14 | 280                            | 49 | 285 | 66 | 347 | 105 |
| onnx/ViT-L/14 | 426                            | 41 | 636 | 58 | 488 | 91 |

### Performance
| Model Name | Text-to-image score (single-label) | Text-to-image score (double-label) | Text-to-image (trible-label) | Image-to-text score | Image-to-Image score |
|---|---|---|---|---|---|
| Vit-B/32 | 92.5 | 78.75 | 46.7 | 91 | good |
| Vit-L/14 | 97.5 | 82.5 | 52.3 | 91 | good |
| fast/Vit-B/32 | 97.5 | 72.5 | 48 | 88 | good |
| fast/Vit-L/14 | 90 | 81.25 | 52.3 | 88 | good |
| openclip/Vit-L/14 | 97.5 | 82.5 | 52.3 | 91 | good |
| opencv/Vit-L-14 | 90 | 81.25 | 52.3 | 88 | good |
| onnx/ViT-L/14 | 97.5 | 82.5 | 52.3 | 91 | good |
> *Vit-B/32 and Vit-L/14 are openai implementations of clip. 
> 
> **fast means the model is using opencv preprocessing and using onnx model to inference

Fastclip, with opencv preprocessing and onnx model, can reduce the preprocessing time of model __ViT-B/32__ without losing performance.

However, onnx model is even increasing the inference time for __ViT-L/14__

Opencv will affect the performance a littile bit but the results are still acceptable.

## Preprocessing:

This section compares different image preprocessing methods.

| TRANSFORMS | TIME (ms) | PROCESSED DIFF (mean) | ENCODE DIFF (mean) |
|:---:|:---:|:---:|:---:|
| original_clip | 14.6 | 0.0 | 0.0 |
| our_clip_implementation | 14.7 | 0.0 | 0.0 |
| opencv_based | 4.67 | 1.22 | 0.19 |
| script_based | 8.07 | 0.037 | 0.0526 |
| rgb_conversion | 12.1 | 0.031 | 0.0475 |
| grey_conversion | 5.33 | 0.053 | 0.121 |
| read_from_cv | 0.940 | 1.22 | 0.19 |


## Inference:

| Models | Time cost | Comments |                                                                         Links                                                                         | Difference |
|:---:|:---:|:---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------:|:---:|
| ViT-B/32 | 7.76 ms ± 127 µs | N/A |                                                                          N/A                                                                          | N/A |
| onnx/ViT-B/32 | 4.16 ms ± 152 µs | Using clip_onnx package |                                                     [link](https://github.com/Lednik7/CLIP-ONNX)                                                      | 9e-6 |
| open_clip/ViT-B-32/openai | 8.05 ms ± 104 µs | N/A |                                                                          N/A                                                                          | N/A |
| Pytorch Dynamic Quantization | N/A | Does not support GPU (support CPU) |                                  [link](https://discuss.pytorch.org/t/does-dynamic-quantization-support-gpu/119231)                                   | N/A |
| Neural Magic | N/A | Does not support GPU (support CPU) |                                                      [link](https://github.com/neuralmagic/docs)                                                      | N/A |
| DeepSpeed | N/A | Can’t get it work on my windows |                                                    [link](https://github.com/microsoft/DeepSpeed)                                                     | N/A |
| Optimized onnx | 4.12 ms ± 152 µs | No difference between onnx | [link](https://github.com/microsoft/onnxruntime/blob/433f262dd551e79f6b3af6d777b5c94eb907622a/onnxruntime/python/tools/transformers/optimizer.py#L53) | 9e-6 |




