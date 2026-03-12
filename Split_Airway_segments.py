import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import skimage.measure as measure
import cv2
import os
import scipy

def keep_largest_connected(label):
    cd, num = measure.label(label, return_num=True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    large_cd = ndimage.binary_fill_holes(large_cd)
    return large_cd.astype(np.uint8)

def loc_trachea_end(mask_array):
    z, y, x = mask_array.shape
    gt_voi = np.where(mask_array > 0)
    _, mask_zmax = min(gt_voi[0]), max(gt_voi[0])
    z_start = mask_zmax-15
    flag = True
    largest_area = 0
    previous_xc = 0
    first_flag = True
    while flag:
        slice_n = mask_array[z_start, :, :]
        slice_n = scipy.ndimage.binary_fill_holes(slice_n).astype(np.uint8)
        contours, _ = cv2.findContours(slice_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) <=2:
            z_start -= 1
            largest_area = max([cv2.contourArea(x) for x in contours])
            index = np.argmax([cv2.contourArea(x) for x in contours])
            xcenter = (max(contours[index][..., 0])[0] + min(contours[index][..., 0])[0]) // 2
            previous_xc = xcenter
            continue
        else:
            contour_areas = [cv2.contourArea(x) for x in contours]
            contour_areas = np.array(contour_areas)
            sorted_area = np.sort(contour_areas)
            center_contour = contours[np.argmax(contour_areas)]
            xcenter = (max(center_contour[..., 0])[0] + min(center_contour[..., 0])[0]) // 2
            # print(xcenter, z_start)
            if largest_area==0:
                largest_area = sorted_area[-1]
                previous_xc = xcenter
            elif sorted_area[-2] < largest_area/2.5:
                if abs(previous_xc-xcenter) > 4:
                    flag=False
                    continue
                z_start -= 1
                largest_area = sorted_area[-1]
                previous_xc = xcenter
                continue
            else:
                flag = False
    return z_start

def remove_trachea(aw, trachea_end_slice):
    gt_voi = np.where(aw > 0)
    _, mask_zmax = min(gt_voi[0]), max(gt_voi[0])
    for i in range(trachea_end_slice, mask_zmax):
        slice_n = aw[i, :, :]
        contours, _ = cv2.findContours(slice_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        contour_areas = np.array(contour_areas)
        cv2.fillPoly(aw[i,...], [contours[np.argmax(contour_areas)]], 2)
    aw_wotrachea = aw.copy()
    aw_wotrachea[aw==2] = 0
    return aw_wotrachea, aw

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC



target_path = "path of prediction masks saved in nii.gz format"
save_path = "saving path"
fids = os.listdir(target_path)

for fid in fids:
    if os.path.exists(save_path+fid):
        print("existing")
        continue
    print("processing: ", fid)
    image_array = sitk.GetArrayFromImage(sitk.ReadImage(target_path + fid))
    aw = keep_largest_connected(image_array)
    # ===================== get aw center ==========================
    z_center = loc_trachea_end(aw) + 5
    slice_n = aw[z_center, :, :]
    contours, _ = cv2.findContours(slice_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    contour_areas = np.array(contour_areas)
    center_contour = contours[np.argmax(contour_areas)]
    xcenter, ycenter = (max(center_contour[..., 0])[0] + min(center_contour[..., 0])[0]) // 2, \
                       (max(center_contour[..., 1])[0] + min(center_contour[..., 1])[0]) // 2
    print("x,y,z centers are: ", xcenter, ycenter, z_center)
    aw_wotrachea, aw_split = remove_trachea(aw, z_center + 2)
    Z_p, Y_p = np.sum(aw_wotrachea, axis=0), np.sum(aw_wotrachea, axis=1)
    Z_p[Z_p > 0] = 1
    Y_p[Y_p > 0] = 1

    Z_p = getLargestCC(Z_p).astype(np.uint8)
    Y_p = getLargestCC(Y_p).astype(np.uint8)

    ZP_xcenter = (np.max(np.where(Z_p > 0)[1]) + np.min(np.where(Z_p > 0)[1])) // 2
    ZP_xcenter = int((ZP_xcenter + xcenter) * 0.5)
    zp_left = Z_p[:, :ZP_xcenter]
    zp_Label = np.zeros_like(Z_p)
    zpL_y, _ = np.where(zp_left > 0)

    adjust_y = (np.max(zpL_y) + np.min(zpL_y)) // 2
    Dis_zpLeft = np.zeros_like(zp_left, np.int16)

    for i in range(zp_left.shape[0]):
        for j in range(zp_left.shape[1]):
            if zp_left[i, j] != 0:
                Dis_zpLeft[i, j] = np.sqrt(np.square(i - ycenter) + np.square(j - zp_left.shape[1]))
    flatten_dis = Dis_zpLeft.reshape(-1)
    refined_dis = np.setdiff1d(flatten_dis, [0])

    sorted = np.sort(refined_dis)
    step1 = np.percentile(sorted, 25)
    step2 = np.percentile(sorted, 55)
    zp_Left_label = np.ones_like(zp_left) * 2
    zp_Left_label[Dis_zpLeft < step1] = 1
    zp_Left_label[Dis_zpLeft == 0] = 0
    zp_Left_label[Dis_zpLeft > step2] = 3

    zp_right = Z_p[:, ZP_xcenter:]
    zpR_y, _ = np.where(zp_right > 0)
    adjust_y = (np.max(zpR_y) + np.min(zpR_y)) // 2
    Dis_zpRight = np.zeros_like(zp_right, np.int16)

    for i in range(zp_right.shape[0]):
        for j in range(zp_right.shape[1]):
            if zp_right[i, j] != 0:
                Dis_zpRight[i, j] = np.sqrt(np.square(i - ycenter) + np.square(j - 0))
    flatten_dis = Dis_zpRight.reshape(-1)
    refined_dis = np.setdiff1d(flatten_dis, [0])

    sorted = np.sort(refined_dis)
    step1 = np.percentile(sorted, 25)
    step2 = np.percentile(sorted, 55)
    zp_Right_label = np.ones_like(zp_right) * 2
    zp_Right_label[Dis_zpRight < step1] = 1
    zp_Right_label[Dis_zpRight == 0] = 0
    zp_Right_label[Dis_zpRight > step2] = 3

    zp_Label[:, :ZP_xcenter] = zp_Left_label
    zp_Label[:, ZP_xcenter:] = zp_Right_label
    Final_label = np.zeros_like(aw_wotrachea, np.uint8)
    Yp_voi = np.where(Y_p > 0)
    zmax, zmin = np.max(Yp_voi[0]), np.min(Yp_voi[0])
    zrange = range(zmin, zmax)
    z_step1 = np.percentile(zrange, 30)
    z_step2 = np.percentile(zrange, 75)
    z, y, x = aw_wotrachea.shape
    for i in range(z):
        for j in range(y):
            for m in range(x):
                if aw_wotrachea[i, j, m] == 0:
                    continue
                if zp_Label[j, m] == 1:
                    if i > z_step2 or i < z_step1:
                        if np.abs(j + m - xcenter - ycenter) > 60:
                            Final_label[i, j, m] = 3
                        else:
                            Final_label[i, j, m] = 1
                    else:
                        Final_label[i, j, m] = 1
                elif zp_Label[j, m] > 1:
                    l1 = zp_Label[j, m]
                    if i > z_step2:
                        l2 = 3
                    elif i > z_step1 and i < z_step2:
                        l2 = 2
                    else:
                        l2 = 3
                    Final_label[i, j, m] = max(l1, l2)
    Final_label[aw_split == 2] = 4
    sitk.WriteImage(sitk.GetImageFromArray(Final_label), save_path+fid)