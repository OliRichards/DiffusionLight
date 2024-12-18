from relighting.environment_map_generator import EnvironmentMapGenerator
from PIL import Image

if __name__ == "__main__":
    generator = EnvironmentMapGenerator()
    generator.load_model()
    image = Image.open("example/bed.png")
    generator(image, 30)