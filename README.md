# fast-clip-research

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


# EC2 Instance
### Add_Documents() Time

| Model Name | Image Indexing Time (CBS = 100) | Text Indexing Time (CBS = 100) | Image Indexing Time (CBS = 50) | Text Indexing Time (CBS = 50) | Image Indexing Time (CBS = 10) | Text Indexing Time (CBS = 10) | Image Indexing Time (CBS = 1) | Text Indexing Time (CBS = 1) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Vit-B/32 * | 18 | 7 | 19 | 8 | 26 | 14 | 70 | 65 |
| Vit-L/14 | 74 | 9 | 74 | 11 | 80 | 15 | 129 | 65 |
| fast/Vit-B/32 ** | 17 | 6 | 36 | 8 | 44 | 14 | 80 | 80 |
| fast/Vit-L/14 | 58 | 9 | 410 | 10 | 420 | 28 | 500 | 139 |
| openclip/Vit-L/14 | 76 | 11.8 | 78 | 13 | 89 | 22 | 220 | 14 |
| opencv/Vit-L-14 | 73 | 9 | 77 | 11 | 88 | 15 | 218 | 65 |
| onnx/ViT-L/14 | 64 | 9 | 60 | 10 | 71 | 28 | 226 | 139 |

For __onnx/ViT-L/14__, there is a converging process in the processing speed. The indexing time starts from 150m/per doc and converges to 64ms/per doc after 40 batches.

