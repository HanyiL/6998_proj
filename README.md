## 6998_proj

###  A description of the project



### A description of the repository



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