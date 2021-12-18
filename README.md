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
   6998_proj python train.py --cnn_arc = 'res152'\ # the pre-trained CNN architecture
   													--model_path = 'where you want to store the trained model'
   ```

4. Then you can input an image to generate captions by the model you have trained

   ```shell
   6998_proj python generateCaption.py --cnn_arc='vgg11'\
                      									--image='coco_test/COCO_val2014_000000000661.jpg'\
                      									--encoder_path='models_vgg11/encoder-5-3000.ckpt'\
                      									--decoder_path='models_vgg11/decoder-5-3000.ckpt'
   ```


### Results (including charts/tables) and your observations 

| Choice of CNN | ![img](file:////private/var/folders/n7/gyf3brbd5z1_l27hgh24508m0000gn/T/com.kingsoft.wpsoffice.mac/wps-hanyili/ksohtml/wpsTG3dpk.png) **<COCO_val2014_000000000074.jpg>** | ![img](file:////private/var/folders/n7/gyf3brbd5z1_l27hgh24508m0000gn/T/com.kingsoft.wpsoffice.mac/wps-hanyili/ksohtml/wpsTY6NFP.png) **<COCO_val2014_000000000544.jpg>** | ![img](file:////private/var/folders/n7/gyf3brbd5z1_l27hgh24508m0000gn/T/com.kingsoft.wpsoffice.mac/wps-hanyili/ksohtml/wpsMtXrWb.png) **<COCO_val2014_000000000661.jpg>** |      |
| :-----------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
|   Resnet18    | a white dog laying on the ground next to a bench             | a baseball player is swinging a bat at a ball                | a hot dog with a pickle on it                                |      |
|   Resnet34    | a dog is sitting on a bench in the street                    | a baseball player holding a bat on a field                   | a sandwich with meat and cheese on a plate                   |      |
|   Resnet50    | a dog sitting on a sidewalk next to a bike                   | a baseball player holding a bat on a field                   | a hot dog with a bun and a side of fries                     |      |
|   Resnet101   | a dog is standing on a sidewalk with a leash                 | a baseball player is getting ready to throw a ball           | a hot dog with mustard and a bun                             |      |
|   Resnet152   | a dog walking on a sidewalk next to a sidewalk               | a baseball player holding a bat on a field                   | a hot dog with mustard and ketchup on it                     |      |

###   

| Choice of CNN | ![img](file:////private/var/folders/n7/gyf3brbd5z1_l27hgh24508m0000gn/T/com.kingsoft.wpsoffice.mac/wps-hanyili/ksohtml/wpsVwVVQO.png) **<COCO_val2014_000000000074.jpg>** | ![img](file:////private/var/folders/n7/gyf3brbd5z1_l27hgh24508m0000gn/T/com.kingsoft.wpsoffice.mac/wps-hanyili/ksohtml/wpsOMT7B2.png) **<COCO_val2014_000000000544.jpg>** | ![img](file:////private/var/folders/n7/gyf3brbd5z1_l27hgh24508m0000gn/T/com.kingsoft.wpsoffice.mac/wps-hanyili/ksohtml/wpswRxWYm.png) **<COCO_val2014_000000000661.jpg>** |
| :-----------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|     VGG11     | a man riding a bike down a street next to a building         | a baseball player swinging a bat at a ball                   | a hot dog with relish and mustard on a plate                 |
|     VGG13     | a dog is sitting on a bench outside                          | a baseball player is swinging at a pitch                     | a sandwich with a pickle and a pickle on a plate             |
|     VGG16     | a dog is sitting on a bench in the shade                     | a baseball player swinging a bat at a ball                   | a hot dog with mustard and ketchup on a bun                  |
|     VGG19     | a dog is sitting on a bench in the street                    | a man in a white uniform holds a baseball bat                | a sandwich with meat and cheese on a plate                   |

| Choice of CNN | BLEU1 | BLEU2 | BLEU3 | BLEU4 |
| :-----------: | ----- | ----- | ----- | ----- |
|   Resnet18    | 0.153 | 0.328 | 0.247 | 0.198 |
|   Resnet34    | 0.154 | 0.329 | 0.247 | 0.198 |
|   Resnet50    | 0.154 | 0.327 | 0.246 | 0.197 |
|   Resnet101   | 0.152 | 0.329 | 0.248 | 0.198 |
|   Resnet152   | 0.152 | 0.328 | 0.247 | 0.198 |

| Choice of CNN | BLEU1 | BLEU2 | BLEU3 | BLEU4 |
| :-----------: | ----- | ----- | ----- | ----- |
|     VGG11     | 0.156 | 0.330 | 0.248 | 0.199 |
|     VGG13     | 0.149 | 0.325 | 0.245 | 0.197 |
|     VGG16     | 0.150 | 0.324 | 0.245 | 0.196 |
|     VGG19     | 0.154 | 0.329 | 0.247 | 0.198 |

#### Observations

- Different versions of same model differs greatly with the number of parameters, they generate image captioning performance with slightly difference on evaluation metric.
- Performance does not improve with the increase in the number of layers(model complexity)
- Different CNN architectures performance almost equally well on evaluation metric
- Most models generate reasonable captions, but the generated captions have great variation