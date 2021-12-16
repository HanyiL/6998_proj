## 6998_proj

###  A description of the project
Image caption generation refers to using ML model to learn how to produce a single sentence for an image. The model needs to learn the content and semantic of the image and translate them into a sentence. In this project we evlauated the performance of different variations of Resnet and VGG models for this task. The performance is evaluated with BLEU score. The evaluation framework is designed to be generic so that other models can also be plugged in easily to get evaluated.

### A description of the repository
Files:
- data_loader.py: download the coco dataset and do initial process of the data.
- preprocess.py: preprocess vocabulary and images including random corp, horizontal flip, normalize, resize, discard less frequent words, etc.
- model.py: the encoder and decoder model configuration.
- train.py: training controller.
- generateCaption.py: generate sample images and result on test data.
- eval.py: calculate BLEU score.

Folders:
- data: the downloaded coco dataset.
- models_x: saved trained models.
- res_captions: geenrated captions.

### Prerequisites

- Natural Language Toolkit (nltk)
- Matplotlib
- NumPy
- Argparse
- Pillow

### Example commands to execute the code     

1. Fetch and download the MSCOCO dataset

   ```shell
   ~ git clone https://github.com/pdollar/coco.git
   ~ cd coco/PythonAPI/
   coco/PythonAPI/ make
   coco/PythonAPI/ python setup.py build
   coco/PythonAPI/ python setup.py install
   cd ~
   ~ git clone https://github.com/HanyiL/6998_proj.git
   cd 6998_proj
   6998_proj mkdir data
   6998_proj wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data/
   6998_proj wget http://images.cocodataset.org/zips/train2014.zip -P ./data/
   6998_proj wget http://images.cocodataset.org/zips/val2014.zip -P ./data/
   6998_proj unzip ./data/captions_train-val2014.zip -d ./data/
   6998_proj rm ./data/captions_train-val2014.zip
   6998_proj unzip ./data/train2014.zip -d ./data/
   6998_proj rm ./data/train2014.zip 
   6998_proj unzip ./data/val2014.zip -d ./data/ 
   6998_proj rm ./data/val2014.zip 
   ```

2. Build vocabulary and resize the train dataset image

   ```shell
   6998_proj python preprocess.py
   ```

3. Train the CNN+LSTM model with specific pre-trained CNN, for example you want to train with Resnet152

   ```shell
   6998_proj python train.py --CNN_model = 'res152'
   ```

4. Then you can input an image to generate captions 

   ```shell
   6998_proj python generateCaption.py --image = './data/val2014/COCO_val2014_000000000042.jpg'
   ```

5. 

### Results (including charts/tables) and your observations 