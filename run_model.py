from relighting.environment_map_generator import EnvironmentMapGenerator
from PIL import Image
import cv2
import os
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

if __name__ == "__main__":
    generator = EnvironmentMapGenerator()
    generator.load_model()
    image = Image.open("example/bed.png")
    res = generator(image, 30)
    cv2.imwrite("output/res.exr", res.astype(np.float32))