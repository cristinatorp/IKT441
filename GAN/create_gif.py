import imageio
import re
import glob
from tqdm import tqdm

image_folder = "images"
convert = lambda text: int(text) if text.isdigit() else text
alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]


def create_gif():
    gifs = glob.glob("gifs/*.gif")
    gifs.sort(key=alphanum_key)
    last_gif_name = gifs[-1]
    new_gif_name = int(re.split("([0-9]+)", last_gif_name)[1]) + 1
    anim_file = f"gifs/{new_gif_name}.gif"

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(f"{image_folder}/*.png")
        filenames.sort(key=alphanum_key)
        last = -1
        for i,filename in enumerate(tqdm(filenames)):
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Gif created ({anim_file}), with {len(filenames)} frames")


if __name__ == '__main__':
    create_gif()
