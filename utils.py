import numpy as np
from scipy import misc

RY = 15
YG = 6
GC = 4
CB = 11
BM = 13
MR = 6
ncols = sum([RY, YG, GC, CB, BM, MR])

def make_color_wheel():
    """A color wheel or color circle is an abstract illustrative
       organization of color hues around a circle.
       This is for making output image easy to distinguish every
       part.
    """
    # These are chosen based on perceptual similarity
    # e.g. one can distinguish more shades between red and yellow
    #      than between yellow and green

    if ncols > 60:
        exit(1)

    color_wheel = np.zeros((ncols, 3))
    i = 0
    # RY: (255, 255*i/RY, 0)
    color_wheel[i: i + RY, 0] = 255
    color_wheel[i: i + RY, 1] = np.arange(RY) * 255 / RY
    i += RY
    # YG: (255-255*i/YG, 255, 0)
    color_wheel[i: i + YG, 0] = 255 - np.arange(YG) * 255 / YG
    color_wheel[i: i + YG, 1] = 255
    i += YG
    # GC: (0, 255, 255*i/GC)
    color_wheel[i: i + GC, 1] = 255
    color_wheel[i: i + GC, 2] = np.arange(GC) * 255 / GC
    i += GC
    # CB: (0, 255-255*i/CB, 255)
    color_wheel[i: i + CB, 1] = 255 - np.arange(CB) * 255 / CB
    color_wheel[i: i + CB, 2] = 255
    i += CB
    # BM: (255*i/BM, 0, 255)
    color_wheel[i: i + BM, 0] = np.arange(BM) * 255 / BM
    color_wheel[i: i + BM, 2] = 255
    i += BM
    # MR: (255, 0, 255-255*i/MR)
    color_wheel[i: i + MR, 0] = 255
    color_wheel[i: i + MR, 2] = 255 - np.arange(MR) * 255 / MR

    return color_wheel

def mapping_to_indices(coords):
    """numpy advanced indexing is like x[<indices on axis 0>, <indices on axis 1>, ...]
        this function convert coords of shape (h, w, 2) to advanced indices
    
    # Arguments
        coords: shape of (h, w)
    # Returns
        indices: [<indices on axis 0>, <indices on axis 1>, ...]
    """
    h, w = coords.shape[:2]
    indices_axis_2 = list(np.tile(coords[:,:,0].reshape(-1), 2))
    indices_axis_3 = list(np.tile(coords[:,:,1].reshape(-1), 1))
    return [indices_axis_2, indices_axis_3]


def flow_to_color(flow, normalized = True):
    """
    # Arguments
        flow: (h, w, 2) flow[u, v] is (y_offset, x_offset)
        normalized: if is True, element in flow is between -1 and 1, which
                    present to 
    """
    # 创建色环
    color_wheel = make_color_wheel() # (55, 3)
    h, w = flow.shape[:2]
    # 需要选取合适的函数来映射flow到色环的索引
    # 这里选择了atan2(-v, -u), 为什么要取负?
    rad = np.sum(flow ** 2, axis = 2) ** 0.5 # shape: (h, w)
    rad = np.concatenate([rad.reshape(h, w, 1)] * 3, axis = -1)
    a = np.arctan2(-flow[:,:,1], -flow[:,:,0]) / np.pi # shape: (h, w) range: (-1, 1)
    fk = (a + 1.0) / 2.0 * (ncols - 1) # -1~1 mapped to 1~ncols
    # 概括:
    # y,x两方向位移差越大，色环上越靠两侧(红色)
    # y,x两方向位移差越小，色环上越靠中间(蓝绿色)

    # 索引要求是整数,这里取了ceil(防了溢出，color_wheel[0]和color_wheel[-1]颜色差不多)和floor
    # 再通过加权求和把这个误差弥补回来
    k0 = np.floor(fk).astype(np.int)
    k1 = (k0 + 1) % ncols
    f = (fk - k0).reshape((-1, 1))
    f = np.concatenate([f] * 3, axis = 1)
    # k0的shape (h, w), 每个元素的值代表了color_wheel上的索引
    color0 = color_wheel[list(k0.reshape(-1))] / 255.0
    color1 = color_wheel[list(k1.reshape(-1))]/ 255.0
    res = (1 - f) * color0 + f * color1
    res = np.reshape(res, (h, w, 3)) # flatten to h*w

    mask = rad <= 1
    res[mask] = (1 - rad * (1 - res))[mask] # increase saturation with radius
    res[~mask] *= .75 # out of range

    return res

if __name__ == '__main__':
    # color_wheel = make_color_wheel()
    h = 100
    w = 100
    flow1 = np.arange(h*w).reshape((h,w,1))
    flow2 = np.arange(h*w - 1, -1, -1).reshape((h,w,1))
    flow = np.concatenate([flow1, flow2], axis = -1)
    color = flow_to_color(flow)
    print(flow)
    misc.imsave('converted.png', color)