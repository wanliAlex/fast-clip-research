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

|         Model Name |           Image Indexing Time |           Text Indexing Time | Text-to-Image score | Image-to-Image score | Image-to-text score
|-------------------:|------------------------------:|-----------------------------:| -----------: | -----------: | -----------: |
|         Vit-B/32   |                          58ms |                         33ms | 92.5 |  good| 91 |
|           Vit-L/14 |              320ms $\pm$ 20ms |                         45ms | 97.5 |  good| 91 |
|      fast/Vit-B/32 |                          36ms |                         20ms | 97.5 |  good| 88 |
|      fast/Vit-L/14 |              410ms $\pm$ 20ms |                         40ms | 90 | good | 88|
|  openclip/Vit-L/14 |              300ms $\pm$ 20ms |               42ms $\pm$ 3ms |97.5 | good |91|
|    opencv/Vit-L-14 |              290ms $\pm$ 20ms |                         42ms |90 | good |88|
|      onnx/ViT-L/14 |              410ms $\pm$ 20ms |                         40ms |97.5| good |91|

> *Vit-B/32 and Vit-L/14 are openai implementations of clip. 
> 
> **fast means the model is using opencv preprocessing and using onnx model to inference

Fastclip, with opencv preprocessing and onnx model, can reduce the preprocessing time of model __ViT-B/32__ without losing performance.

However, onnx model is even increasing the inference time for __ViT-L/14__

Opencv will affect the performance a littile bit but the results are still acceptable.
