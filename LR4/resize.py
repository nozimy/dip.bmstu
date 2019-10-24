import os
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io, filters, color, draw

SehunFaceFolder = 'Sehun/face'
StanleyWeberFaceFolder = 'StanleyWeber/face'

SehunTrainingFolder = 'Training/Sehun'
SehunTestingFolder = 'Testing/Sehun'

StanleyWeberTrainingFolder = 'Training/StanleyWeber'
StanleyWeberTestingFolder = 'Testing/StanleyWeber'


def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img = io.imread(os.path.join(folder,filename))
            if img is not None:
#                img = color.rgb2gray(img)
                images.append(img)
        return images
    

def my_resize(images):
    width = 200
    height = 200
    for idx, img in enumerate(images):
        image_resized = resize(img, (width, height), anti_aliasing=True)
        images[idx]=image_resized
    return images

def save_images(images, folder):
    for idx, img in enumerate(images):
        io.imsave(os.path.join(folder,str(idx)+'.png'),img)
    
sehun_imgs = load_images_from_folder(SehunFaceFolder)
sehun_imgs = my_resize(sehun_imgs)
save_images(sehun_imgs, SehunTrainingFolder)

weber_imgs = load_images_from_folder(StanleyWeberFaceFolder)
weber_imgs = my_resize(weber_imgs)
save_images(weber_imgs, StanleyWeberTrainingFolder)
