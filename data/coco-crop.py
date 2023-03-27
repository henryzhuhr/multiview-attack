import os
from pycocotools.coco import COCO

ROOT = os.path.split(os.path.split(__file__)[0])[0]
dataDir = os.path.join(ROOT , 'data/coco')
dataType = 'val2017'
annotation_file = f'{dataDir}/annotations/ovd_ins_{dataType}_b.json'


print(dataDir)
coco=COCO(annotation_file)

cats = coco.loadCats(coco.getCatIds())