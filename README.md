# ColonGPT
## Introduction
- A vision-language model based on the LLaVA architecture, capable of image captioning, classification, and object detection for colonoscopy.
- Utilizes LLaMA, SigLIP, and Spatial Pyramid Pooling (SPP) for improved visual understanding, assisting clinicians in anomaly detection and generating automated reports.

## Demo

## Dataset
the dataset and original work is from [https://github.com/ai4colonoscopy/IntelliScope](https://github.com/ai4colonoscopy/IntelliScope)

## Setup:
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

## train:
The model was trained on two RTX 6000 ADA GPUs

Stage 1 training:
```
./src/train/run_train_stage1.sh
```

Stage 2 training:
```
./src/train/run_train_stage2.sh
```
