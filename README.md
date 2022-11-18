# fast-clip-research


| Model Name     | Image Indexing Time|  Text Indexing Time | Text-to-Image score | Image-to-Image score | Image-to-text score
| -------------: | -----------: |-----------: | -----------: | -----------: | -----------: |
| Vit-B/32  *   |  58ms       |  33ms | 92.5 |  good| 91 |
| Vit-L/14   *  |  320ms $\pm$ 20ms       |  45ms | 97.5 |  good| 91 |
| fast/Vit-B/32 **    |  36ms      |  20ms | 97.5 |  good| 88 |
| fast/Vit-L/14  **| 410ms $\pm$ 20ms | 40ms | 90 | good | 88|
| openclip/Vit-L/14 | 300ms $\pm$ 20ms | 42ms $\pm$ 3ms  |97.5 | good |91|
| opencv/Vit-L-14 | 290ms $\pm$ 20ms | 42ms |90 | good |88|


>_ * Vit-B/32 and Vit-L/14 are openai implementations of clip. 
>_ ** fast means the model is using opencv preprocessing and using onnx model to inference
