# ColGPT
## Introduction
- A vision-language model based on the LLaVA architecture for image classification, image captioning, and object detection for colonoscopy.
- Utilizes LLaMA, SigLIP, and Spatial Pyramid Pooling (SPP) for improved visual understanding, assisting clinicians in anomaly detection and generating automated reports.

## Demo
https://github.com/user-attachments/assets/90c0f120-55d8-456e-9fe7-ebe6e36f82ca

## Performance
| Task             |   F1   |  UAR  | Mean IoU | Accuracy IoU | Perplexity |
|------------------|--------|-------|----------|--------------|------------|
| Classification   |84.3    |85.0   |N/A       |N/A           |1.4         |
| Object Detection |91.4    |92.8   |54.7      |65.1          |5.56        |
| Image Captioning |N/A     |N/A    |N/A       |N/A           |4.86        |

## Dataset
the dataset and original work is from [https://github.com/ai4colonoscopy/IntelliScope](https://github.com/ai4colonoscopy/IntelliScope)

## Setup
- Install the dependencies:
  ```
  git clone git@github.com:wilnn/ColGPT.git
  cd ColGPT
  conda create -n ColGPT python=3.12.11
  pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu129
  pip install -r requirements.txt
  pip install flash-attn==2.8.0.post2 --no-build-isolation
  ```
  note that running `pip install flash-attn==2.8.0.post2 --no-build-isolation` can take a very long time to finish. you can skip it if you want.  

- Then, you need to obtain the dataset from the original repository above and put the dataset into the directory ColGPT/dataset. It should look like this:
  ```
  ├──ColGPT/dataset
              ├──ColonINST
                      ├──Json-file
                      ├──Positive-images
  ```

- Finally, you run this to clean and fix the dataset:
  ```
  python ./src/EDA_and_preprocessing.py
  ```
  Now, the dataset directory should look like this:
 ```
  ├──ColGPT/dataset
              ├──ColonINST
                      ├──Json-file
                      ├──Json-file-clean
                      ├──Positive-images
  ```

## train
The model was trained on two RTX 6000 ADA GPUs

**Stage 1 training:**
```
./src/train/run_train_stage1.sh
```

**Stage 2 training for 3 tasks:**
```
./src/train/run_train_stage2_3tasks.sh
```

**Stage 2 training for image captioning:**
```
./src/train/run_train_stage2_cap.sh
```

## Acknowledgment
The original work is from [https://github.com/ai4colonoscopy/IntelliScope](https://github.com/ai4colonoscopy/IntelliScope)
