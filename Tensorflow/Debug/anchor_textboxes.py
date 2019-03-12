import numpy as np

anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7
aspect_ratios = [1, 2, 3, 5, 7, 10]
#self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
num_anchors = len(aspect_ratios)* 2 #* len(self.scale_ratios)

input_shape = np.array([608, 608])

def _get_anchor_hw():
    anchor_hw = []
    for s in anchor_areas:
        for ar in aspect_ratios:  # w/h = ar
            anchor_h = np.sqrt(s/ar)
            anchor_w = ar * anchor_h
            anchor_hw.append([anchor_h, anchor_w])

    num_fms = len(anchor_areas)
    anchor_hw = np.array(anchor_hw)
    return anchor_hw.reshape(num_fms, -1, 2)

def _get_anchor_boxes():

    anchor_hw = _get_anchor_hw()
    num_fms = len(anchor_areas)
    fm_sizes = [np.ceil(input_shape/pow(2.,i+3)) for i in range(num_fms)]  # p3 -> p7 feature map sizes
    print(fm_sizes)

    boxes = []
    for i in range(num_fms):
        fm_size = fm_sizes[i]
        grid_size = input_shape / fm_size

        fm_h, fm_w = int(fm_size[0]), int(fm_size[1]) # fm_h == fm_w : True
        fm_w = fm_w * 2 #- 1 #for add vertical offset

        meshgrid_x = ((np.arange(0, fm_w) / 2) + 0.5) * grid_size[0]
        meshgrid_y = (np.arange(0, fm_h) + 0.5) * grid_size[1]
        meshgrid_x, meshgrid_y = np.meshgrid(meshgrid_x, meshgrid_y)

        yx = np.vstack((meshgrid_y.ravel(), meshgrid_x.ravel())).transpose()
        yx = np.tile(yx.reshape((fm_h, fm_w, 1, 2)), (6, 1))

        hw = np.tile(anchor_hw[i].reshape(1, 1, 6, 2), (fm_h, fm_w, 1, 1))
        box = np.concatenate([yx, hw], 3)  # [y,x,h,w]
        boxes.append(box.reshape(-1,4))

    return np.concatenate(boxes, 0)
    #return tf.cast(tf.concat(boxes, 0), tf.float32)


anchor_hw = _get_anchor_hw()
print(anchor_hw)
print("=" * 40)

anchor_boxes = _get_anchor_boxes()
print(anchor_boxes.shape)

