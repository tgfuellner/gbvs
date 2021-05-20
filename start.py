#!/usr/bin/python3

from saliency_models import gbvs, ittikochneibur
import cv2
import time, os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import argh
from tqdm import tqdm

def display(img, gbvs, ittikochneibur):
    fig = plt.figure(figsize=(10, 3))

    fig.add_subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.gca().set_title("Original Image")
    plt.axis('off')

    fig.add_subplot(1, 3, 2)
    plt.imshow(gbvs, cmap='gray')
    plt.gca().set_title("GBVS")
    plt.axis('off')

    fig.add_subplot(1, 3, 3)
    plt.imshow(ittikochneibur, cmap='gray')
    plt.gca().set_title("Itti Koch Neibur")
    plt.axis('off')

    plt.show()


@argh.arg('images', type=open, nargs="+", help="source image(s)")
@argh.arg("-s", "--show", help="show maps", action="store_true")
def main(images, show=False):
    """Creates saliency and ittikochneibur maps"""

    pbar = tqdm(images)
    for img in pbar:
        imgName = img.name
        pbar.set_description("processing {}".format(img.name))

        img.close()
        img = cv2.imread(imgName)

        saliency_map_gbvs = gbvs.compute_saliency(img)
        saliency_map_ikn = ittikochneibur.compute_saliency(img)

        imgNameWithoutExtension = os.path.splitext(imgName)[0]

        oname = "{}_gbvs.png".format(imgNameWithoutExtension)
        cv2.imwrite(oname, saliency_map_gbvs)
        norm = np.interp(saliency_map_gbvs, (saliency_map_gbvs.min(), saliency_map_gbvs.max()), (0, 1))
        df = pd.DataFrame(data=norm)
        df.to_csv("{}_gbvs.csv".format(imgNameWithoutExtension), index=False, header=False)

        oname = "{}_ittikochneibur.png".format(imgNameWithoutExtension)
        cv2.imwrite(oname, saliency_map_ikn)
        norm = np.interp(saliency_map_ikn, (saliency_map_ikn.min(), saliency_map_ikn.max()), (0, 1))
        df = pd.DataFrame(data=norm)
        df.to_csv("{}_ittikochneibur.csv".format(imgNameWithoutExtension), index=False, header=False)

        if show:
            display(img, saliency_map_gbvs, saliency_map_ikn)


argh.dispatch_command(main)