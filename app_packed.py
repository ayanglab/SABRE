import sys
import os
import csv
from glob import glob
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QLabel,
    QCheckBox, QTextEdit, QLineEdit,
    QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import QThread, pyqtSignal
import torch
import skimage.measure as measure
import numpy as np
import cv2
from lungmask import LMInferer
import scipy
import SimpleITK as sitk
from models.FuzzyAttentionModel import FuzzyAttention_3DUNet

def get_resource_path(relative_path):
    """获取资源文件的绝对路径，兼容开发环境和打包环境"""
    try:
        # PyInstaller 打包后的临时文件夹路径
        base_path = sys._MEIPASS
    except Exception:
        # 开发环境路径
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def adjust_window(image, window_center=-300, window_width=1800):
    win_min = window_center - window_width // 2
    win_max = window_center + window_width // 2
    image = 255.0 * (image - win_min) / (win_max - win_min)
    image[image>255] = 255
    image[image<0] = 0
    return image


def zcore_normalization(images, mean=None, std=None):
    if mean==None or std==None:
        raise Exception("compute the mean and std of training set first!")
    images = (images - mean)/std
    return images


def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def large_connected_domain(label):
    cd, num = measure.label(label, return_num=True, connectivity=1)
    if num ==0:
        return scipy.ndimage.binary_fill_holes(label).astype(np.uint8)
    else:
        volume = np.zeros([num])
        for k in range(num):
            volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
        volume_sort = np.argsort(volume)
        flag = True
        idex = -1
        while flag:
            label = (cd == (volume_sort[idex] + 1)).astype(np.uint8)
            label_voi = np.where(label > 0)
            z_min, z_max = min(label_voi[0]), max(label_voi[0])
            y_min, y_max = min(label_voi[1]), max(label_voi[1])
            x_min, x_max = min(label_voi[2]), max(label_voi[2])
            z, y, x = z_max-z_min, y_max-y_min,x_max-x_min
            if z/y > 10 or z/x>10:
                print("check this prediction!!!")
                idex += 1
            else:
                flag = False
        label = scipy.ndimage.binary_fill_holes(label)
        label = label.astype(np.uint8)
    return label


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

def post_trachea(image_array, mask_array):
    z, y, x = mask_array.shape
    gt_voi = np.where(mask_array > 0)
    _, mask_zmax = min(gt_voi[0]), max(gt_voi[0])
    z_start = mask_zmax - 30
    flag = True
    Failed_flag = False
    while flag:
        slice_n = mask_array[z_start, :, :]
        contours, _ = cv2.findContours(slice_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            z_start += 1
            continue
        contour_area = cv2.contourArea(contours[0])
        if contour_area < 50:
            z_start += 1
        else:
            flag = False
        if z_start >= z:
            print("give up post-processing")
            Failed_flag = True
            break
    if Failed_flag:
        return mask_array
    else:
        z_end = z
        gt_sub = mask_array[z_start:z_end, :, :]
        img_sub = image_array[z_start:z_end, :, :]

        gtsub_voi = np.where(gt_sub > 0)
        roi_voxels = img_sub[gtsub_voi]

        airway_mean = np.mean(roi_voxels)
        airway_std = np.std(roi_voxels)
        airway_min = airway_mean - airway_std * 2
        airway_max = airway_mean + airway_std * 2

        y_min, y_max = min(gtsub_voi[1]) - 10, max(gtsub_voi[1]) + 10
        x_min, x_max = min(gtsub_voi[2]) - 5, max(gtsub_voi[2]) + 5
        for i in range(z_start, z_end):
            slice = image_array[i, y_min:y_max, x_min:x_max]
            slice_mask = mask_array[i, y_min:y_max, x_min:x_max]
            threshmask = np.zeros_like(slice_mask)
            threshmask[slice > airway_min] = 1
            threshmask[slice > airway_max] = 0
            mask_array[i, y_min:y_max, x_min:x_max] += threshmask

        mask_array[mask_array > 0] = 1
        mask_array = large_connected_domain(mask_array)
        return mask_array


def safe_write_sitk(img: sitk.Image, out_path: str, ref_img: sitk.Image = None) -> str:
    if ref_img is not None:
        img.CopyInformation(ref_img)
    try:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(out_path)
        writer.UseCompressionOn()
        writer.Execute(img)
        return out_path
    except Exception as e1:
        try:
            root, ext = os.path.splitext(out_path)
            if ext == ".gz":
                root, _ = os.path.splitext(root)
            alt_path = root + ".nii"
            writer = sitk.ImageFileWriter()
            writer.SetFileName(alt_path)
            writer.UseCompressionOff()
            writer.Execute(img)
            return alt_path
        except Exception as e2:
            alt2_path = root + ".mha"
            writer = sitk.ImageFileWriter()
            writer.SetFileName(alt2_path)
            writer.UseCompressionOff()
            writer.Execute(img)
            return alt2_path


def run_segmentation(progress, net, mask_model, case_path, use_enhanced, save_path, device):
    """
    Dummy segmentation function.
    Replace with your actual segmentation code.
    """
    os.makedirs(save_path, exist_ok=True)

    # Load image (DICOM folder or NIfTI file)
    if os.path.isdir(case_path):
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesFileNames(case_path)
        if not series:
            raise ValueError(f"No DICOM series found in folder: {case_path}")
        reader.SetFileNames(series)
        image = reader.Execute()
        fid = os.path.basename(case_path)
    else:
        image = sitk.ReadImage(case_path)
        if image is None or image.GetSize() == (0, 0, 0):
            raise ValueError(f"Invalid NIfTI file: {case_path}")
        fid = os.path.basename(case_path).split('.nii')[0]
    image_array = sitk.GetArrayFromImage(image)
    final_prediction = np.zeros_like(image_array)
    z, y, x = image_array.shape

    if z < 100:
        raise ValueError('Requires HRCT with >100 slices')

    if y!=512 and x!=512:
        raise ValueError('The slide size should be 512*512')
    progress.emit("Start lung lobes modelling...")
    lung = mask_model.apply(image).astype(np.uint8)
    if lung is None:
        raise RuntimeError("Lungmask failed to return a prediction. Check model path and packaging.")
    progress.emit(f"lung mask extracted!")

    lung_img = sitk.GetImageFromArray(lung)
    lung_img.CopyInformation(image)
    if not os.path.exists(os.path.join(save_path, fid)):
        os.makedirs(os.path.join(save_path, fid))
    lung_out =  os.path.join(save_path, fid, 'lung.nii.gz')
    saved_lung_path = safe_write_sitk(image, lung_out, ref_img=image)
    progress.emit(f"Saved lung mask to: {saved_lung_path}")
    # sitk.WriteImage(lung_img, os.path.join(save_path, fid, 'lung.nii.gz'))

    # Crop to lung region
    gt_voi = np.where(lung)
    z_min, z_max = min(gt_voi[0]), z
    y_min, y_max = min(gt_voi[1]), max(gt_voi[1])
    x_min, x_max = min(gt_voi[2]), max(gt_voi[2])
    vol = image_array[z_min:z_max, y_min:y_max, x_min:x_max]

    progress.emit("Start airway modelling...")

    # Preprocessing
    vol = adjust_window(vol, window_center=-300, window_width=1800)
    vol = zcore_normalization(vol, mean=124.6, std=64)

    cube = (128, 96, 144)
    pred = np.zeros(vol.shape, dtype=np.float32)
    count = np.zeros_like(pred)
    z, y, x = vol.shape
    for i in range(0, z, cube[0] // 2):
        for j in range(0, y, cube[1] // 2):
            for k in range(0, x, cube[2] // 2):
                z_start_idex = min(i, z - cube[0])
                z_end_idex = min(z, i + cube[0])
                x_start_idex = min(k, x - cube[2])
                x_end_idex = min(x, k + cube[2])
                y_start_idex = min(j, y - cube[1])
                y_end_idex = min(y, j + cube[1])
                patch = vol[z_start_idex:z_end_idex, y_start_idex:y_end_idex, x_start_idex:x_end_idex]
                inp = torch.from_numpy(patch[np.newaxis, np.newaxis, ...]).float().to(device)
                with torch.no_grad():
                    y_out = net(inp).cpu().detach().numpy()
                if y_out.shape[1] != 1:
                    y_out = np.round(y_out).squeeze()[-1, ...]
                elif y_out.shape[1] == 1:
                    y_out = np.round(y_out).squeeze()
                pred[z_start_idex:z_end_idex, y_start_idex:y_end_idex, x_start_idex:x_end_idex] += y_out
                count[z_start_idex:z_end_idex, y_start_idex:y_end_idex, x_start_idex:x_end_idex] += np.ones(cube)
    pred = pred / count
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = np.squeeze(pred)
    pred = large_connected_domain(pred)
    final_prediction[z_min:z_max, y_min:y_max, x_min:x_max] = pred
    # Enhanced post-processing
    if use_enhanced:
        try:
            final_prediction = post_trachea(image_array, final_prediction)
        except Exception:
            pass

    # Save airway mask
    final_prediction = final_prediction.astype(np.uint8)
    out_img = sitk.GetImageFromArray(final_prediction)
    out_img.CopyInformation(image)
    sitk.WriteImage(out_img, os.path.join(save_path, fid, 'airway.nii.gz'))

    # ********************************* Airway Split Algorithm ****************************************
    progress.emit("Splitting airway into branches...")
    z_center = loc_trachea_end(final_prediction) + 5
    slice_n = final_prediction[z_center, :, :]
    contours, _ = cv2.findContours(slice_n, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    contour_areas = np.array(contour_areas)
    center_contour = contours[np.argmax(contour_areas)]
    xcenter, ycenter = (max(center_contour[..., 0])[0] + min(center_contour[..., 0])[0]) // 2, \
                       (max(center_contour[..., 1])[0] + min(center_contour[..., 1])[0]) // 2

    aw_wotrachea, aw_split = remove_trachea(final_prediction, z_center + 2)
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
                        if np.abs(j + m - xcenter - ycenter) > (60 / 512) * image_array.shape[1]:
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
    sitk.WriteImage(sitk.GetImageFromArray(Final_label), os.path.join(save_path, fid, 'awsplits.nii.gz'))
    return Final_label, final_prediction, lung


def compute_metrics(aw_pd, lung, rescale_factor):
    """
    Dummy metrics computation.
    Replace with your actual metrics code.
    """
    lung = np.sum(lung)

    radius_volumeList = np.array([0, 0, 0, 0])  # 0-2.2-4,4-8,>8
    radius_volumeList[0] = np.sum(aw_pd == 3)
    radius_volumeList[1] = np.sum(aw_pd == 2)
    radius_volumeList[2] = np.sum(aw_pd == 1)
    radius_volumeList[3] = np.sum(aw_pd == 4)

    awv_nor_lung = radius_volumeList / lung

    # f3d = fractal_dimension_3D(aw_array)
    STotalAV = awv_nor_lung.sum() * rescale_factor
    STermAV = awv_nor_lung[0] * rescale_factor
    SSmallAV = awv_nor_lung[1] * rescale_factor
    SMedAV = awv_nor_lung[2] * rescale_factor
    SPAV = STermAV+SSmallAV
    return  [STotalAV, STermAV, SSmallAV, SMedAV, SPAV]


class Worker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str, str)

    def __init__(self, input_folder, output_folder, use_enhanced, use_dicom, rescale):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.use_enhanced = use_enhanced
        self.use_dicom = use_dicom
        self.rescale = rescale
        self._is_running = True

    def stop(self):
        """调用这个函数来停止线程"""
        self._is_running = False

    def run(self):
        # Prepare file list
        if self.use_dicom:
            files = [os.path.join(self.input_folder, e) for e in os.listdir(self.input_folder)]
        else:
            files = glob(os.path.join(self.input_folder, '*.nii')) + glob(os.path.join(self.input_folder, '*.nii.gz'))
        total = len(files)
        if total == 0:
            self.progress.emit('No files found.')
            self.finished.emit()
            return

        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.progress.emit('GPU detected, using gpu to inference.')
        else:
            device = torch.device('cpu')
            self.progress.emit('GPU not detected, using cpu to inference.')

        model_path = get_resource_path("models/best_model.pt")
        mask_model_path = get_resource_path("models/unet_r231-d5d2fc3d.pth")

        if not os.path.exists(model_path):
            self.error.emit("Model Error", f"Model file not found: {model_path}")
            return
        if not os.path.exists(mask_model_path):
            self.error.emit("Model Error", f"Mask model file not found: {mask_model_path}")
            return

        try:
            net = torch.load(model_path, map_location=device).to(device)
            net.eval()
            mask_model = LMInferer(modelname='R231', modelpath=mask_model_path)
            if mask_model is None:
                raise RuntimeError(f"Failed to load lungmask model from {mask_model_path}")
            self.progress.emit('Model loaded successfully.')
        except Exception as e:
            self.error.emit("Model Loading Error", str(e))
            return

        # Prepare CSV
        os.makedirs(self.output_folder, exist_ok=True)
        csv_path = os.path.join(self.output_folder, 'metrics.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['CaseID', 'STotalAV', 'STermAV', 'SSmallAV', 'SMedAV', 'SPAV'])

            for idx, f in enumerate(files, 1):
                if not self._is_running:
                    self.progress.emit("Processing stopped by user.")
                    break
                case_id = os.path.basename(f)
                try:
                    self.progress.emit(f'[{idx}/{total}] ======================== Processing {case_id} ========================')
                    aw_split, aw_pd, lung = run_segmentation(self.progress, net, mask_model, f, self.use_enhanced, self.output_folder, device)
                    lung = (lung > 0).astype(np.uint8)
                    self.progress.emit("Calculating SABRE-based metrics...")
                    metrics = compute_metrics(aw_split, lung, self.rescale)
                    writer.writerow([case_id] + metrics)
                    self.progress.emit(f'   ✔ Done {case_id}')
                except Exception as e:
                    self.error.emit(case_id, str(e))
                    self.progress.emit(f'   ✖ Error on {case_id}: {e}')

        self.progress.emit('All files processed.')
        self.finished.emit()


class SimpleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.input_folder = ''
        self.output_folder = ''
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('SABRE V1.0')
        self.setGeometry(300, 300, 600, 500)

        self.input_label = QLabel('Input Folder: Not selected')
        self.input_btn = QPushButton('Select Input Folder')
        self.input_btn.clicked.connect(self.select_input)

        self.output_label = QLabel('Output Folder: Not selected')
        self.output_btn = QPushButton('Select Output Folder')
        self.output_btn.clicked.connect(self.select_output)

        self.enhanced_cb = QCheckBox('Use Post-Processing')
        self.dicom_cb = QCheckBox('Process DICOM Files')

        self.rescale_label = QLabel('Rescale Factor:')
        self.rescale_input = QLineEdit('100.0')
        self.rescale_input.setFixedWidth(80)

        opts_layout = QHBoxLayout()
        opts_layout.addWidget(self.enhanced_cb)
        opts_layout.addWidget(self.dicom_cb)
        opts_layout.addWidget(self.rescale_label)
        opts_layout.addWidget(self.rescale_input)
        opts_layout.addStretch()

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)

        self.run_btn = QPushButton('Run')
        self.run_btn.clicked.connect(self.start_processing)

        self.stop_btn = QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.input_btn)
        path_layout.addWidget(self.output_btn)

        label_layout = QHBoxLayout()
        label_layout.addWidget(self.input_label)
        label_layout.addWidget(self.output_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()

        vbox = QVBoxLayout()
        vbox.addLayout(path_layout)
        vbox.addLayout(label_layout)
        vbox.addLayout(opts_layout)
        vbox.addWidget(self.log_widget)
        vbox.addLayout(btn_layout)

        self.setLayout(vbox)

    def select_input(self):
        folder = QFileDialog.getExistingDirectory(self,'Select Input Folder')
        if folder:
            self.input_folder = folder
            self.input_label.setText(f'Input Folder: {folder}')

    def select_output(self):
        folder = QFileDialog.getExistingDirectory(self,'Select Output Folder')
        if folder:
            self.output_folder = folder
            self.output_label.setText(f'Output Folder: {folder}')

    def processing_finished(self):
        self.log_widget.append('Processing complete or stopped.')
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)   #

    def start_processing(self):
        if not self.input_folder or not self.output_folder:
            self.log_widget.append('Please select both input and output folders.')
            return
        try:
            rescale = float(self.rescale_input.text())
        except ValueError:
            self.log_widget.append('Invalid rescale factor.')
            return
        use_enhanced = self.enhanced_cb.isChecked()
        use_dicom = self.dicom_cb.isChecked()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_widget.clear()
        self.worker = Worker(self.input_folder, self.output_folder, use_enhanced, use_dicom, rescale)
        self.worker.progress.connect(self.update_log)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()

    # === 新增：停止函数 ===
    def stop_processing(self):
        if self.worker is not None:
            try:
                self.worker.stop()
                self.log_widget.append("Stop signal sent.")
            except Exception as e:
                self.log_widget.append(f"Failed to stop: {e}")

    def update_log(self,message):
        self.log_widget.append(message)
        self.log_widget.verticalScrollBar().setValue(self.log_widget.verticalScrollBar().maximum())

    def handle_error(self,filename,error_str):
        print(f'Error on {filename}: {error_str}')

    def processing_finished(self):
        self.log_widget.append('Processing complete.')
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = SimpleApp()
    window.show()
    sys.exit(app.exec_())