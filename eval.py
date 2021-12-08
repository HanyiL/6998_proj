import pycocotools
from pycocotools.coco import COCO
annFile='data/annotations/captions_val2014.json'
coco = COCO(annFile)