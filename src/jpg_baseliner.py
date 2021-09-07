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
    for y in range(int(height*frm), int(height*to)):
        alpha = 255-int((y - height*frm)/height/(to - frm) * 255)
        for x in range(width):
            pixels[x, y] = pixels[x, y][:3] + (alpha,)
    for y in range(y, height):
        for x in range(width):
            pixels[x, y] = pixels[x, y][:3] + (0,)

    return im

if __name__ == "__main__":
    im = Image.open('test.jpg')
    im2 = alphaV(im, 0.3, 0.7)
    im2.show()
    #im2.save('birdfade.png')

    pass
