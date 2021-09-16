import sys
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt

# prograssive vertical alpha fading from @frm% to @to%
def alphaV(im, frm, to):
    assert to >= frm
    assert to > 0 and to <= 1
    assert frm >= 0 and frm < 1
    width, height = im.size

    im.putalpha(255)

    pixels = im.load()
    for y in range(int(height*frm), int(height*to)):
        alpha = 255-int((y - height*frm)/height/(to - frm) * 255)
        for x in range(width):
            pixels[x, y] = pixels[x, y][:3] + (alpha,)
    for y in range(y, height):
        for x in range(width):
            pixels[x, y] = pixels[x, y][:3] + (0,)

    return im

# prograssive horizontal alpha fading from @frm% to @to%
def alphaH(im, frm, to):
    assert to >= frm
    assert to > 0 and to <= 1
    assert frm >= 0 and frm < 1
    width, height = im.size

    im.putalpha(255)

    pixels = im.load()
    for x in range(int(width*frm), int(width*to)):
        alpha = 255-int((x - width*frm)/width/(to - frm) * 255)
        for y in range(height):
            pixels[x, y] = pixels[x, y][:3] + (alpha,)
    for x in range(x, width):
        for y in range(height):
            pixels[x, y] = pixels[x, y][:3] + (0,)

    return im

def overlay(bim, fim):
    return Image.alpha_composite(bim, fim)

def alphaVmix(im1, im2, alpha):
    im2.putalpha(255)
    alphaV(im1, alpha/2, alpha)
    return overlay(im2,im1)

def alphaHmix(im1, im2, alpha):
    im2.putalpha(255)
    alphaH(im1, alpha/2, alpha)
    return overlay(im2,im1)

def removeAlpha(img):
    noalpha = Image.new("RGB", img.size, (255, 255, 255))
    noalpha.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    return noalpha

def hmix(im1, im2, alpha):
    return removeAlpha(alphaHmix(im1, im2, alpha))

def vmix(im1, im2, alpha):
    return removeAlpha(alphaVmix(im1, im2, alpha))

# get ith image from cifar
def get_image(i):

    assert(i < 60000)
    findex = 1+(i // 10000)
    fname = ''
    if findex > 5:
        fname = 'test_batch'
    else:
        fname = 'data_batch_' + str(findex)

    f = open('../data/cifar/' + fname, 'rb')

    data_dict = pickle.load(f, encoding='bytes')

    f.close()

    images = data_dict[b'data']

    single_img = np.array(images[i % 10000])

    single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))

    return single_img_reshaped


def squarize(list, shape=(32,32,3)):
    pi = 1
    for d in shape:
        pi *= d

    assert(len(list) == d)
    return list.reshape(shape)


def linearize(matrix):

    shape = matrix.shape
    pi = 1
    for d in shape:
        pi *= d
    return matrix.reshape(pi)


if __name__ == "__main__":

    img1 = get_image(32000)
    img2 = get_image(34000)

    pImage1 = Image.fromarray(img1)
    pImage2 = Image.fromarray(img2)

    #pImage = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

    pImage = alphaVmix(pImage2,pImage1,0.7)

    removeAlpha(pImage1).save('fie1.jpg', 'JPEG', quality=80)
    removeAlpha(pImage2).save('fie2.jpg', 'JPEG', quality=80)
    #pImage.show()
    removeAlpha(pImage).save('fie.jpg', 'JPEG', quality=80)

    """
    im1 = Image.open('bird.jpg')
    im2 = Image.open('ship.jpg')

    im3 = alphaVmix(im2,im1,0.7)
    #im3.show()

    removeAlpha(im3).save('foo.jpg', 'JPEG', quality=80)
    """
