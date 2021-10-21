import numpy as np

def interpolate(a, b, f, threshold):
    c = np.array(a + f * (b - a), int)
    mag = np.linalg.norm(c - a)

    return c if (mag > threshold) else a

def reverseCamera(img, b):
    if (b): return np.array([i[::-1] for i in img])
    else: return img

def getCenterPos(position, size):
    return (int(position[0] + size[0] // 2), int(position[1] + size[1] // 2))

def getTopLeftPos(position, size):
    return (int(position[0] - size[0] // 2), int(position[1] - size[1] // 2))

def cropImage(img, start, end):
    sx, sy = start
    ex, ey = end
    ow, oh = img.shape[:2][::-1]
    w, h = (ex - sx, ey - sy)
    
    if (sx < 0):
        sx = 0; ex = w

    if (sy < 0):
        sy = 0; ey = h

    if (ex > ow):
        ex = ow; sx = ow - w

    if (ey > oh):
        ey = oh; sy = oh - h

    return img[sy:ey, sx:ex]

def changeRatio(size, ratio):
    w, h = size
    rw, rh = ratio

    if (h / rh * rw > 0):
        # change width
        return (int(h / rh * rw), h)
    else:
        # change height
        return (w, int(w / rw * rh))

def changeImageRatio(img, ratio):
    w, h = img.shape[:2][::-1]
    outW, outH = changeRatio((w, h), ratio)

    startX = (outW - w) // 2
    startY = (outH - h) // 2

    output = np.full((outH, outW, 3), 0, np.uint8)
    output[startY:startY + h, startX:startX + w] = img

    return output