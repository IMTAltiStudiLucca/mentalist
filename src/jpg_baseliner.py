import sys
import numpy as np
from PIL import Image

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


if __name__ == "__main__":
    im1 = Image.open('bird.jpg')
    im2 = Image.open('ship.jpg')

    im3 = alphaVmix(im2,im1,0.7)
    #im3.show()

    removeAlpha(im3).save('foo.jpg', 'JPEG', quality=80)
