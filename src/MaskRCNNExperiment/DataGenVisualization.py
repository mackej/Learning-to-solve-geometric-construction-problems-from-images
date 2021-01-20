from MaskRCNNExperiment import GeometryDataset
from mrcnn import visualize
import numpy as np
import pickle
import lzma
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--import_file", default="datagen_val_data", type=str, help="Which file should be imported")
    args = parser.parse_args()
    with open(args.import_file, "rb") as data_gen_file:
        data_gen = pickle.load(data_gen_file)


    def get_ax(rows=1, cols=1, size=8):
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax
    image_ids = np.random.choice(len(data_gen.image_info), 10)
    for image_id in image_ids:
        image = data_gen.load_image(image_id)
        mask, class_ids = data_gen.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, data_gen.class_names)
