import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from collections import defaultdict
import os
from skimage.morphology import binary_dilation
from sklearn import metrics as sklearn_metrics

import matplotlib.pyplot as plt

# from models.banger.filter import LocalMasker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RADIUS_MIN = 0.12 / 0.01225
EPSILON = 1e-5
PIX2ARC = 0.01225


def get_result_thresh(
    detection_map,
    labels_gt,
    labels_real,
    threshold,
    radius,
    # sep_max,
    plot=True,
):
    """
    Args:
    -----
    * detection_map: (H, W)
    * labels_gt, with a label for each source: (H, W)
    * threshold: float
    * radius: float
    * n_gt: number of sources gt, float
    * sep_max: maximal separation, float

    Returns:
    --------
    * tp, fp, fn: int
    """
    assert detection_map.ndim == 2
    assert labels_gt.ndim == 2
    assert labels_real.ndim == 2
    H, W = detection_map.shape
    # detection_map = detection_map.clone()
    # image_max = ndi.maximum_filter(detection_map, size=20, mode='constant')

    yy, xx = np.mgrid[:H, :W]
    tp = 0
    fp = 0
    all_x = []
    all_y = []
    status_real = []
    status_tp = []
    status_fp = []
    binary_gt = labels_gt > 0
    # print(f"{threshold=}")
    # while True:
    # print("local_max ..")
    # print("a")
    # print(f"{threshold=}")
    coords = peak_local_max(
        detection_map,
        # min_distance=2 * radius,
        # min_distance=1,
        min_distance=radius,
        # min_distance=1,
        threshold_abs=threshold,
        # footprint=footprint,
        # exclude_border=True,
    )

    coords_x = coords[:, 1]
    coords_y = coords[:, 0]
    values = detection_map[coords_y, coords_x]
    idx = np.argsort(values)[::-1]
    # breakpoint()
    plot = False
    # if plot:
    # fig, axes = plt.subplots(ncols=len(idx), nrows=2)

    for i in idx:
        y = coords_y[i]
        x = coords_x[i]
        value = detection_map[y, x]
        if value == -np.inf:
            continue

        yy_c = yy - y
        xx_c = xx - x
        rr = np.sqrt(yy_c**2 + xx_c**2)
        mask = rr < radius
        detection_map[mask] = -np.inf

        all_x.append(x)
        all_y.append(y)

        is_real = labels_real[y, x] > 0
        if is_real:
            status_real.append(True)
        else:
            status_real.append(False)

        is_tp = labels_gt[y, x] > 0

        if is_tp:
            tp += 1
            labels_gt[labels_gt == labels_gt[y, x]] = 0
            status_tp.append(True)
            status_fp.append(False)
        else:
            fp += 1
            status_tp.append(False)
            status_fp.append(True)
        # if plot:
        # plot = (np.abs(threshold - 3.93) < 0.02)
        # plot = plot | (np.abs(threshold - 3.68) < 0.02)
        plot1 = np.abs(threshold - 1.94) < 0.2
        plot2 = np.abs(threshold - 2.01) < 0.2
        # if plot1 or plot2:
        if plot:
            print("Plotting..")
            print()
            print(f"{threshold=}")
            print(f"{tp=}")
            print(f"{fp=}")
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(detection_map)
            ax[0].scatter(x, y, c="r")
            ax[1].imshow(binary_gt)
            ax[1].scatter(x, y, c="r")
            plt.show()
            breakpoint()
            # axes[0, i].imshow(detection_map)
            # axes[0, i].plot(x, y, "r.")
            # axes[1, i].imshow(binary_gt)
            # axes[1, i].plot(x, y, "r.")
    # if plot:
    # plt.show()

    coords = {
        "coords_x": np.array(all_x),
        "coords_y": np.array(all_y),
        "status_tp": np.array(status_tp),
        "status_fp": np.array(status_fp),
        "status_real": np.array(status_real),
    }

    return tp, fp, coords


def get_result_thresh_2(
    detection_map,
    labels_gt,
    labels_real,
    threshold,
    radius,
    # sep_max,
    plot=False,
):
    """
    Args:
    -----
    * detection_map: (H, W)
    * labels_gt, with a label for each source: (H, W)
    * threshold: float
    * radius: float
    * n_gt: number of sources gt, float
    * sep_max: maximal separation, float

    Returns:
    --------
    * tp, fp, fn: int
    """
    assert detection_map.ndim == 2
    assert labels_gt.ndim == 2
    assert labels_real.ndim == 2
    H, W = detection_map.shape
    # detection_map = detection_map.clone()
    # image_max = ndi.maximum_filter(detection_map, size=20, mode='constant')

    yy, xx = np.mgrid[:H, :W]
    tp = 0
    fp = 0
    all_x = []
    all_y = []
    status_real = []
    status_tp = []
    status_fp = []
    binary_gt = labels_gt > 0
    # print(f"{threshold=}")
    while True:
        # print("local_max ..")
        coords = peak_local_max(
            detection_map,
            # min_distance=2 * radius,
            min_distance=1,
            threshold_abs=threshold,
            # footprint=footprint,
            # exclude_border=True,
        )
        # print("local_max done")
        # print(coords)
        if len(coords) == 0:
            break
        # breakpoint()
        coords_x = coords[:, 1]
        coords_y = coords[:, 0]
        values = detection_map[coords_y, coords_x]
        argmax = np.argmax(values)
        y = coords_y[argmax]
        x = coords_x[argmax]
        yy_c = yy - y
        xx_c = xx - x
        rr = np.sqrt(yy_c**2 + xx_c**2)
        mask = rr < radius
        detection_map[mask] = -np.inf

        all_x.append(x)
        all_y.append(y)

        is_real = labels_real[y, x] > 0
        if is_real:
            status_real.append(True)
        else:
            status_real.append(False)

        is_tp = labels_gt[y, x] > 0

        if is_tp:
            tp += 1
            labels_gt[labels_gt == labels_gt[y, x]] = 0
            status_tp.append(True)
            status_fp.append(False)
        else:
            fp += 1
            status_tp.append(False)
            status_fp.append(True)
        if plot:
            print("Plotting..")
            print()
            print(f"{threshold=}")
            print(f"{tp=}")
            print(f"{fp=}")
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(detection_map)
            ax[0].plot(x, y, "r.")
            ax[1].imshow(binary_gt)
            ax[1].plot(x, y, "r.")
            plt.show()

    coords = {
        "coords_x": np.array(all_x),
        "coords_y": np.array(all_y),
        "status_tp": np.array(status_tp),
        "status_fp": np.array(status_fp),
        "status_real": np.array(status_real),
    }

    return tp, fp, coords


def get_result_thresh_old(
    detection_map,
    labels_gt,
    labels_real,
    threshold,
    radius,
    sep_max,
    plot=False,
):
    """
    Args:
    -----
    * detection_map: (H, W)
    * labels_gt, with a label for each source: (H, W)
    * threshold: float
    * radius: float
    * n_gt: number of sources gt, float
    * sep_max: maximal separation, float

    Returns:
    --------
    * tp, fp, fn: int
    """
    assert detection_map.ndim == 2
    assert labels_gt.ndim == 2
    assert labels_real.ndim == 2
    H, W = detection_map.shape
    # detection_map = detection_map.clone()
    # image_max = ndi.maximum_filter(detection_map, size=20, mode='constant')

    n_gt = len(np.unique(labels_gt)) - 1
    # print(f"n_gt={n_gt}")
    binary_gt = labels_gt > 0
    binary_real = labels_real > 0

    # dmap = detection_map.copy()
    # dmap[dmap < threshold] = 0
    # dmap[binary_real] = 0

    coords = peak_local_max(
        detection_map,
        min_distance=2 * radius,
        threshold_abs=threshold,
        p_norm=2,
        # exclude_border=True,
    )
    coords_x = coords[:, 1]
    coords_y = coords[:, 0]
    if sep_max is not None:
        x_c = coords_x - W // 2
        y_c = coords_y - H // 2
        rr = np.sqrt(y_c**2 + x_c**2)
        mask_sep = rr < (sep_max * 1.0 / PIX2ARC)
        coords_x = coords_x[mask_sep]
        coords_y = coords_y[mask_sep]
        coords = coords[mask_sep, :]

    n_pos = len(coords_x)

    status_real = binary_real[coords_y, coords_x]
    n_pos_real = np.sum(status_real)

    status_tp = binary_gt[coords_y, coords_x] & (~status_real)
    # status_fp = binary_gt[coords_y, coords_x] == 0
    status_fp = (binary_gt[coords_y, coords_x] == 0) & (~status_real)
    # status_fp = binary_gt[coords_y, coords_x] == 0

    # n_pos_synth = n_pos - n_pos_real
    tp = np.sum(status_tp)
    # fp = n_pos_synth - tp
    fp = np.sum(status_fp)
    assert fp >= 0
    # if fp > 0:
    # breakpoint()
    # if fp < 0:
    # breakpoint()
    fn = n_gt - tp
    coords = {
        "coords_x": coords_x,
        "coords_y": coords_y,
        "status_tp": status_tp,
        "status_fp": status_fp,
        "status_real": status_real,
    }

    if plot:
        print()
        print(f"{threshold=}")
        print(f"{tp=}")
        print(f"{fp=}")
        print(f"{fn=}")
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(detection_map)
        ax[0].plot(coords_x, coords_y, "r.")
        ax[1].imshow(binary_gt)
        ax[1].plot(coords_x, coords_y, "r.")
        plt.show()

    # mask already detected

    dmap = detection_map.copy()
    mask = np.zeros((H, W), dtype=bool)
    for (
        y,
        x,
    ) in zip(coords_y, coords_x):
        label = labels_gt[y, x]
        labels_gt[labels_gt == label] = 0
        yy, xx = np.mgrid[:H, :W]
        yy -= y
        xx -= x
        rr = np.sqrt(yy**2 + xx**2)
        mask[rr < radius] = True

    dmap[mask] = -np.inf

    return tp, fp, fn, coords, dmap


def compute_distance_map(n_sources, coords, size, coords_shift):
    yy, xx = np.mgrid[:size, :size]
    # coords = coords.numpy()
    dist_glob = np.ones((size, size)) * np.inf
    markers = np.zeros((size, size), dtype=int)
    for i in range(n_sources):
        c_y, c_x = coords[i] + coords_shift
        dist = np.sqrt((yy - c_y) ** 2 + (xx - c_x) ** 2)
        dist_glob = np.minimum(dist, dist_glob)
        markers[int(np.round(c_y)), int(np.round(c_x))] = i + 1
    return dist_glob, markers


def compute_labels_real(n_sources, coords, size, radius):
    if n_sources == 0:
        return np.zeros((size, size), dtype=bool)
    yy, xx = np.mgrid[:size, :size]
    # coords = coords.numpy()
    dist_glob = np.ones((size, size)) * np.inf
    markers = np.zeros((size, size), dtype=int)
    mask_glob = np.zeros((size, size), dtype=int)
    for i in range(n_sources):
        _radius = radius[i]
        c_y, c_x = coords[i]
        dist = np.sqrt((yy - c_y) ** 2 + (xx - c_x) ** 2)
        mask = dist < _radius
        if np.sum(mask > 0):
            dist_glob = np.minimum(dist, dist_glob)
            marker_x = np.clip(int(np.round(c_x)), 0, size - 1)
            marker_y = np.clip(int(np.round(c_y)), 0, size - 1)
            markers[marker_y, marker_x] = i + 1
            mask_glob |= mask

    # mask_glob = dist < radius
    labels = watershed(-dist, markers, mask=mask_glob)
    # if np.sum(labels) > 0:
    # breakpoint()
    return labels


def compute_labels_gt(n_sources, coords, size, coords_shift, radius):
    if n_sources == 0:
        return np.zeros((256, 256))
    dist, markers = compute_distance_map(
        n_sources=n_sources,
        coords=coords,
        size=size,
        coords_shift=coords_shift,
    )
    mask = dist < radius
    labels = watershed(-dist, markers, mask=mask)
    return labels


def compute_labels_pred_watershed(snr, threshold):
    # snr = snr.cpu().numpy()
    mask = snr > threshold

    coords = peak_local_max(snr, footprint=np.ones((3, 3)), labels=mask)
    seed = np.zeros(snr.shape, dtype=bool)
    seed[tuple(coords.T)] = True
    markers, _ = ndi.label(seed)
    labels = watershed(-snr, markers, mask=mask)
    # breakpoint()

    return labels


def compute_labels_pred_old_old(snr, threshold):
    # snr = snr.cpu().numpy()
    mask_threshold = snr > threshold

    coords = peak_local_max(
        snr, footprint=np.ones((3, 3)), labels=mask_threshold
    )
    mask_local_max = np.zeros(snr.shape, dtype=bool)
    mask_local_max[tuple(coords.T)] = True
    mask_final = mask_threshold & mask_local_max
    labels, _ = ndi.label(mask_final)
    # labels = watershed(-snr, markers, mask=mask)
    # breakpoint()

    return labels


def compute_labels_pred_contiguous(snr, threshold):
    # snr = snr.cpu().numpy()
    mask_threshold = snr > threshold

    labels, _ = ndi.label(mask_threshold)
    # labels = watershed(-snr, markers, mask=mask)
    # breakpoint()

    return labels


def compute_iou_matrix(labels_gt, labels_pred):
    # breakpoint()
    set_labels_pred = set(list(labels_pred.flatten()))
    if 0 in set_labels_pred:
        set_labels_pred.remove(0)

    set_labels_gt = set(list(labels_gt.flatten()))
    if 0 in set_labels_gt:
        set_labels_gt.remove(0)

    # n_gt = np.maximum(n_gt, 1)
    # n_pred = np.maximum(n_pred, 1)
    # print(n_gt, n_pred)
    iou_matrix = np.zeros((len(set_labels_gt), len(set_labels_pred)))
    for i, l_pred in enumerate(set_labels_pred):
        mask_pred = labels_pred == l_pred
        for j, l_gt in enumerate(set_labels_gt):
            mask_gt = labels_gt == l_gt
            intersection = np.sum(
                np.logical_and(
                    mask_gt,
                    mask_pred,
                )
            )
            union = np.sum(
                np.logical_or(
                    mask_gt,
                    mask_pred,
                )
            )
            iou = intersection / (union + 1e-5)
            iou_matrix[j, i] = iou
    return iou_matrix


def get_tp_fp_fn(iou_matrix, iou_threshold=0):
    n_gt, n_pred = iou_matrix.shape
    if (n_gt == 0) & (n_pred == 0):
        return 0, 0, 0
    elif (n_gt == 0) & (n_pred > 0):
        return 0, n_pred, 0
    elif (n_gt > 0) & (n_pred == 0):
        return 0, 0, n_gt

    tp = np.sum(np.max(iou_matrix, axis=1) > iou_threshold)
    fp = np.sum(np.max(iou_matrix, axis=0) <= iou_threshold)
    fn = np.sum(np.max(iou_matrix, axis=1) <= iou_threshold)

    return tp, fp, fn


def get_tp_fp_fn_pix(labels_gt, labels_pred):
    mask_gt = labels_gt > 0
    mask_pred = labels_pred > 0
    tp = np.sum(mask_gt & mask_pred)
    fp = np.sum(mask_pred & (~mask_gt))
    fn = np.sum((~mask_pred) & mask_gt)
    return tp, fp, fn


def sliding_window(radius, values, bins, w_size):
    sum_out = []
    sum_sq_out = []
    count_out = []
    for v_bin in bins:
        rmin = np.maximum(v_bin - w_size / 2, RADIUS_MIN)
        rmax = v_bin + w_size / 2
        mask = (radius > rmin) & (radius < rmax)
        sum_out.append(np.sum(values[mask]))
        sum_sq_out.append(np.sum(values[mask] ** 2))
        count_out.append(np.sum(mask))
    return sum_out, sum_sq_out, count_out


def extract_snr_values(snr, coords, n_sources):
    H, W = snr.shape
    n, _ = coords.shape
    coords_int = np.round(coords[:n_sources]).astype(int)
    snr_coords = snr[coords_int[:, 0], coords_int[:, 1]]
    return snr_coords


class MetricsSuite:
    def __init__(
        self,
        coords_shift,
        snr_min,
        snr_max,
        n_snr,
        radius_gt,
        iou_thresh,
        n_bins,
        window_size,
        save_output,
        verbose,
        labels_pred_method,
        erode,
        run_name,
        sep_max,
        main_threshold,
        max_threshold,
        suffix="",
    ):
        self.coords_shift = coords_shift
        # self.thresholds = thresholds
        self.radius_gt = radius_gt
        self.iou_thresh = iou_thresh
        self.suffix = suffix

        # if self.thresholds is None:
        # self.thresholds = np.linspace(snr_min, snr_max, n_snr)
        # self.thresholds = np.logspace(-1, 2, n_snr)
        # thresholds = list(np.logspace(0, 2, n_snr))
        # thresholds = list(np.logspace(-2, 2, n_snr) - 1e-2)
        # thresholds = list(np.logspace(-3, 2, n_snr) - 1e-3 - 0.1)
        self.n_snr = n_snr
        # thresholds = list(np.logspace(-3, 2, n_snr))
        thresholds = list(np.linspace(0, 1, n_snr))
        # thresholds = list(np.logspace(-3, 2, n_snr))
        self.max_threshold = max_threshold
        if self.max_threshold is not None:
            thresholds = np.array(thresholds)
            thresholds = thresholds * self.max_threshold / np.max(thresholds)
            # eps = 1e-3
            # thresholds = list(np.linspace(-eps, self.max_threshold + eps, n_snr))
            thresholds = list(thresholds)
            # pass
        self.main_threshold = main_threshold
        thresholds.append(self.main_threshold)
        self.thresholds = np.array(sorted(thresholds))
        # self.thresholds = np.array([0, self.main_threshold])
        # print(f"{self.thresholds=}")
        self.erode = erode
        self.run_name = run_name

        # self.auc_new = AUC(kernel_size=self.radius_gt * 2)

        # n_bins = 30

        self.window_size = window_size
        self.radial_bins = np.linspace(RADIUS_MIN, 128, n_bins)

        self.tmp = {
            "sum": np.zeros(n_bins),
            "sum_sq": np.zeros(n_bins),
            "count": np.zeros(n_bins),
        }
        self.save_output = save_output

        self.path_output = os.path.join(
            os.getcwd(), f"output_detection{suffix}.pt"
        )

        if self.save_output:
            self.to_save = defaultdict(list)

        self.verbose = verbose
        self.labels_pred_method = labels_pred_method
        self.sep_max = sep_max
        assert self.labels_pred_method in ["contiguous", "watershed"]
        if self.labels_pred_method == "contiguous":
            self.labels_pred_fn = compute_labels_pred_contiguous
        else:
            self.labels_pred_fn = compute_labels_pred_watershed

        self.all_data = defaultdict(list)

    def add(
        self,
        alpha,
        sigma_alpha,
        n_sources,
        coords_gt,
        alpha_coords,
        n_sources_real,
        coords_real,
        radius_real,
        inference_time,
    ):
        """
        Compute metrics

        Args:
        -----
        * alpha: torch.tensor(b, H, W)
        * sigma_alpha: torch.tensor(b, H, W)
        * mask: torch.tensor(b, H, W)
        * n_sources: (b,)
        * coords: (b, n, 2)
        * n_sources_real: (b,)
        * coords_real: (b, n, 2)
        * radius_real: (b, n)

        Returns:
        --------
        * all_metrics: dict
        """
        snr = alpha / sigma_alpha
        snr = snr.cpu().numpy()
        snr[np.isinf(snr)] = np.nan
        b, H, W = snr.shape
        assert b == 1
        # snr += np.random.rand(H, W) * 1e-3
        mask_nan = np.isnan(snr)
        for _ in range(self.erode):
            mask_nan = binary_dilation(mask_nan)
        snr[mask_nan] = np.nan
        sigma_alpha = sigma_alpha.cpu().numpy()
        n_sources = n_sources.cpu().numpy()
        coords_gt = coords_gt.cpu().numpy()
        n_sources_real = n_sources_real.cpu().numpy()
        coords_real = coords_real.cpu().numpy()
        radius_real = radius_real.cpu().numpy()
        alpha_coords = alpha_coords.cpu().numpy()

        # breakpoint()
        self.all_data["snr"].append(snr[0])
        self.all_data["alpha"].append(alpha[0])
        self.all_data["sigma_alpha"].append(sigma_alpha[0])
        self.all_data["n_sources"].append(int(n_sources))
        self.all_data["coords_gt"].append(coords_gt[0])
        self.all_data["n_sources_real"].append(int(n_sources_real))
        self.all_data["coords_real"].append(coords_real[0])
        self.all_data["radius_real"].append(radius_real[0])
        self.all_data["alpha_coords"].append(alpha_coords[0])
        self.all_data["inference_time"].append(inference_time)

    def get_footprint(self, H, W):
        if self.sep_max is None:
            return np.zeros((H, W), dtype=bool)

        yy, xx = np.mgrid[:H, :W]
        yy_c = yy - H // 2
        xx_c = xx - W // 2
        rr = np.sqrt(yy_c**2 + xx_c**2)
        mask_sep = rr < (self.sep_max * 1.0 / PIX2ARC)
        return mask_sep

    def process(self):
        snr = np.stack(self.all_data["snr"])
        mask_nan = np.isnan(snr)
        snr[mask_nan] = 0
        print(f"NAN: {mask_nan.sum()=}")
        n_maps, H, W = snr.shape
        sigma_alpha = self.all_data["sigma_alpha"]
        n_sources = self.all_data["n_sources"]
        coords_gt = self.all_data["coords_gt"]
        n_sources_real = self.all_data["n_sources_real"]
        coords_real = self.all_data["coords_real"]
        radius_real = self.all_data["radius_real"]
        alpha = self.all_data["alpha"]
        alpha_coords = self.all_data["alpha_coords"]
        inference_time = self.all_data["inference_time"]
        avg_inference_time = np.mean(inference_time)
        # print("\n\n")
        # print(f"{alpha=}")
        # print(f"{coords_gt=}")

        # breakpoint()

        # print("TO CHANGE ///")
        # snr_min = np.min(snr)
        # snr_max = np.max(snr) + 1e-8
        # snr_min = 0
        # snr_min = -0.1
        # snr_min = -0.05
        # snr_min = -0.05
        # snr_max = 20
        # snr_max = 100
        snr_min = np.min(snr)
        # snr_max = np.max(snr) + 1e-8
        # snr_min = -0.05
        snr_max = np.max(snr) + 1e-8
        # snr_max = np.max(snr) + 0.001
        # breakpoint()
        thresholds = np.logspace(-4, 0, self.n_snr)
        # thresholds = np.linspace(0, 1, self.n_snr)
        thresholds = thresholds * (snr_max - snr_min) + snr_min
        thresholds = list(thresholds)
        thresholds.append(self.main_threshold)
        thresholds.append(snr_min)
        thresholds = sorted(thresholds)
        self.thresholds = np.array(thresholds)[::-1]
        n_thresholds = len(self.thresholds)
        # print(f"{self.thresholds=}")
        self.snr_min = snr_min
        self.snr_max = snr_max
        print(f"{self.snr_min=}")
        print(f"{self.snr_max=}")
        print(f"{self.thresholds=}")
        print(f"{avg_inference_time=}")

        self.all_metrics = {
            "n_sources": sum(n_sources),
            "tp": np.zeros(n_thresholds),
            "fp": np.zeros(n_thresholds),
            "snr_coords": [],
            "alpha_coords": [],
            "coords": [],
            "avg_inference_time": avg_inference_time,
        }

        print(f"Processing detection maps.. ({n_maps=})")
        for i in tqdm(range(n_maps)):
            # self.add_radial_contrast(sigma_alpha[i])
            labels_gt = compute_labels_gt(
                n_sources=n_sources[i],
                coords=coords_gt[i],
                size=snr.shape[-1],
                radius=self.radius_gt,
                coords_shift=self.coords_shift,
            )
            # _n_sources_real = n_sources_real[i]
            # _coords_real = coords_real[i, :_n_sources_real]
            # _radius_real = radius_real[i, :_n_sources_real]
            labels_real = compute_labels_real(
                n_sources=n_sources_real[i],
                coords=coords_real[i],
                size=snr.shape[-1],
                radius=radius_real[i],
            )

            snr_coords = extract_snr_values(
                snr=snr[i],
                n_sources=n_sources[i],
                coords=coords_gt[i],
            )
            self.all_metrics["snr_coords"] += list(snr_coords)
            self.all_metrics["coords"] += list(coords_gt[i][: n_sources[i]])
            self.all_metrics["alpha_coords"] += list(
                alpha_coords[i][: n_sources[i]]
            )
            # if _n_sources_real > 0:
            # breakpoint()
            # assert np.sum((labels_real > 0) & (labels_gt > 0)) == 0
            # self.auc_new.add(
                # snr=snr[i], mask_cat=labels_gt > 0, mask_valid=1 - labels_real
            # )
            if not np.sum((labels_real > 0) & (labels_gt > 0)) == 0:
                print("WARNING: OVERLAP BETWEEN SOURCE REAL AND SYNTH")
                # breakpoint()

            footprint = self.get_footprint(H, W)
            dmap = snr[i].copy()
            dmap[~footprint] = -np.inf

            if self.save_output:
                labels_pred = self.labels_pred_fn(
                    snr=snr[i],
                    threshold=self.main_threshold,
                )

                _, _, coords_pred = get_result_thresh(
                    detection_map=dmap.copy(),
                    labels_gt=labels_gt.copy(),
                    labels_real=labels_real.copy(),
                    radius=self.radius_gt,
                    threshold=self.main_threshold,
                )
                coords_y_gt = (
                    coords_gt[i][: n_sources[i], 0] + self.coords_shift
                )
                coords_x_gt = (
                    coords_gt[i][: n_sources[i], 1] + self.coords_shift
                )

                coords_y_real = coords_real[i][: n_sources_real[i], 0]
                coords_x_real = coords_real[i][: n_sources_real[i], 1]
                mask_y = (coords_y_real >= 0) & (coords_y_real < 256)
                mask_x = (coords_x_real >= 0) & (coords_x_real < 256)
                mask_real = mask_x & mask_y
                coords_y_real = coords_y_real[mask_real]
                coords_x_real = coords_x_real[mask_real]

                # labels_pred[labels_real > 0] = 0
                self.to_save["snr"].append(snr[i])
                self.to_save["alpha"].append(alpha[i])
                self.to_save["sigma_alpha"].append(sigma_alpha[i])
                self.to_save["labels_gt"].append(labels_gt)
                self.to_save["labels_pred"].append(labels_pred)
                self.to_save["labels_real"].append(labels_real)
                self.to_save["coords_x_gt"].append(coords_x_gt)
                self.to_save["coords_y_gt"].append(coords_y_gt)
                self.to_save["coords_y_real"].append(coords_y_real)
                self.to_save["coords_x_real"].append(coords_x_real)
                for k, v in coords_pred.items():
                    self.to_save[k].append(v)

            # idx_5 = np.argmin(np.abs(self.thresholds - 5))
            # dmap = snr[i].copy()
            tp_map = []
            fp_map = []
            for j, thresh in enumerate(self.thresholds):
                tp, fp, _ = get_result_thresh(
                    detection_map=dmap,
                    labels_gt=labels_gt,
                    labels_real=labels_real,
                    radius=self.radius_gt,
                    threshold=thresh,
                    plot=False,
                )
                if self.verbose:
                    print(f"idx={i}, {thresh=}, {tp=}, {fp=}, {fn=}")
                tp_map.append(tp)
                fp_map.append(fp)

                # if (j == idx_5) and (fp > 0):
                # breakpoint()

                # debug = False
                # if (fp > 0) and debug:
                # print(iou_matrix)
                # plt.imshow(labels_gt > 0)
                # plt.imshow(labels_pred > 0, alpha=0.5)
                # plt.show()
                # breakpoint()
                # breakpoint()
                # self.all_metrics[f"tp_{thresh}"].append(tp)
                # self.all_metrics[f"fp_{thresh}"].append(fp)
                self.all_metrics["tp"][j] += tp
                self.all_metrics["fp"][j] += fp
                # self.all_metrics[f"fn_{thresh}"].append(fn)
                # self.all_metrics[f"tp_pix_{thresh}"].append(tp_pix)
                # self.all_metrics[f"fp_pix_{thresh}"].append(fp_pix)
                # self.all_metrics[f"fn_pix_{thresh}"].append(fn_pix)

            tp_cum = np.cumsum(tp_map)
            tn_cum = n_sources[i] - tp_cum
            fp_cum = np.cumsum(fp_map)
            tpr = tp_cum / n_sources[i]
            precision = tp_cum / (tp_cum + fp_cum)
            fdr = 1 - precision
            fpr = fp_cum / (fp_cum)
            # plt.plot(fdr, tpr)
            # plt.show()

        if self.save_output:
            torch.save(self.to_save, self.path_output)
            print(f"Output saved to {self.path_output}")
            self.print_path(self.path_output)

    def add_radial_contrast(self, sigma_alpha):
        H, W = sigma_alpha.shape
        # print(H, W)
        yy, xx = np.mgrid[:H, :W]
        center = H // 2
        yy -= center
        xx -= center
        rr = np.sqrt(yy**2 + xx**2)
        # breakpoint()
        s, s_sq, n = sliding_window(
            radius=rr,
            values=sigma_alpha,
            bins=self.radial_bins,
            w_size=self.window_size,
        )
        if torch.tensor(s).isnan().any():
            breakpoint()
        self.tmp["sum"] += s
        self.tmp["sum_sq"] += s_sq
        self.tmp["count"] += n

    def get_radial_contrast(self):
        # print(self.tmp["count"])
        # print(self.radial_bins)
        assert np.all(self.tmp["count"] > 0)
        # breakpoint()
        # print("COUNT:")
        # print(self.tmp["count"])
        mean = self.tmp["sum"] / self.tmp["count"]
        var = self.tmp["sum_sq"] / self.tmp["count"] - mean**2
        # breakpoint()
        return mean, var

    def get_metrics(self):
        # metrics = {"precision": [], "recall": [], "tpr": [], "fpr": []}
        self.process()
        snr = self.all_data["snr"]
        metrics = defaultdict(list)
        n_sources = self.all_metrics["n_sources"]
        tp_cum = np.cumsum(self.all_metrics["tp"])
        fp_cum = np.cumsum(self.all_metrics["fp"])
        mask = (tp_cum > 0) | (fp_cum > 0)
        # breakpoint()
        # auc_new = self.auc_new.process()
        # self.auc_new.reset()

        tp_cum = tp_cum[mask]
        fp_cum = fp_cum[mask]
        thresholds = self.thresholds[mask]

        fn_cum = n_sources - tp_cum

        precision = list(tp_cum / (tp_cum + fp_cum + EPSILON))
        recall = list(tp_cum / (tp_cum + fn_cum + EPSILON))
        # tpr = tp_cum / (tp_cum + fn_cum + EPSILON)
        tpr = list(tp_cum / (n_sources + EPSILON))
        fdr = list(fp_cum / (tp_cum + fp_cum + EPSILON))
        # fdr = list(fp_cum / (tp_cum + fn_cum + EPSILON))

        recall.insert(0, 0.0)
        precision.insert(0, 1.0)
        recall.append(1.0)
        precision.append(0.0)

        tpr.append(1.0)
        fdr.append(1.0)
        tpr.insert(0, 0.0)
        fdr.insert(0, 0.0)

        tpr = np.array(tpr)[::-1]
        fdr = np.array(fdr)[::-1]
        metrics["precision"] = np.array(precision)[::-1]
        metrics["recall"] = np.array(recall)[::-1]
        metrics["tpr"] = tpr
        metrics["fdr"] = fdr
        metrics["tp"] = tp_cum[::-1]
        metrics["fp"] = fp_cum[::-1]

        mask_fdr = fdr == 0
        if np.sum(mask_fdr) == 0:
            tpr_max = 0.0
        else:
            tpr_max = np.max(tpr[mask_fdr])
        metrics["tpr_max"] = tpr_max
        # print(f"{auc_new=}")
        # plt.plot(self.thresholds, fdr)
        # plt.show()
        # breakpoint()

        # for thresh in self.thresholds:
        # # objects
        # tp = np.sum(self.all_metrics[f"tp_{thresh}"])
        # fp = np.sum(self.all_metrics[f"fp_{thresh}"])
        # fn = np.sum(self.all_metrics[f"fn_{thresh}"])
        # n_sources = tp + fn

        # precision = tp / (tp + fp + EPSILON)
        # recall = tp / (tp + fn + EPSILON)
        # tpr = tp / (tp + fn + EPSILON)
        # fpr = fp / (tp + fp + EPSILON)
        # metrics["precision"].append(precision)
        # metrics["recall"].append(recall)
        # metrics["tpr"].append(tpr)
        # metrics["fpr"].append(fpr)

        # # pixels
        # tp_pix = np.sum(self.all_metrics[f"tp_pix_{thresh}"])
        # fp_pix = np.sum(self.all_metrics[f"fp_pix_{thresh}"])
        # fn_pix = np.sum(self.all_metrics[f"fn_pix_{thresh}"])
        # # n_sources_pix = tp_pix + fn_pix

        # precision_pix = tp_pix / (tp_pix + fp_pix + EPSILON)
        # recall_pix = tp_pix / (tp_pix + fn_pix + EPSILON)
        # tpr_pix = tp_pix / (tp_pix + fn_pix + EPSILON)
        # fpr_pix = fp_pix / (tp_pix + fp_pix + EPSILON)
        # metrics["precision_pix"].append(precision_pix)
        # metrics["recall_pix"].append(recall_pix)
        # metrics["tpr_pix"].append(tpr_pix)
        # metrics["fpr_pix"].append(fpr_pix)

        # metrics[f"precision_{thresh}"] = precision
        # metrics[f"recall_{thresh}"] = recall
        metrics = dict(metrics)
        metrics["thresholds"] = thresholds[::-1]
        metrics["n_sources"] = n_sources
        metrics["main_threshold"] = self.main_threshold

        metrics["alpha_coords"] = self.all_metrics["alpha_coords"]
        metrics["coords"] = self.all_metrics["coords"]
        metrics["snr_coords"] = self.all_metrics["snr_coords"]
        metrics["avg_inference_time"] = self.all_metrics["avg_inference_time"]

        # contrast
        # mean, var = self.get_radial_contrast()
        # metrics["contrast"] = {
        # "mean": mean,
        # "var": var,
        # "bins": self.radial_bins,
        # }
        # breakpoint()
        precision_min = np.min(metrics["precision"])
        argmin = np.argmin(metrics["precision"])
        # breakpoint()

        try:
            snr_min = metrics["thresholds"][argmin]
            print(f"precision_min: {precision_min} with snr={snr_min}")
        except Exception as e:
            print(f"WARNING: {e}")
            breakpoint()
        try:
            recall = list(metrics["recall"])
            precision = list(metrics["precision"])
            metrics["auc_pr"] = sklearn_metrics.auc(recall, precision)
            print(f"{metrics['auc_pr']=}")
        except Exception as e:
            print(e)
            metrics["auc_pr"] = -1

        try:
            tpr = list(metrics["tpr"])
            fdr = list(metrics["fdr"])

            auc = sklearn_metrics.auc(
                tpr,
                fdr,
            )
            metrics["auc_fdr_tpr"] = 1 - auc
            print(f"{metrics['auc_fdr_tpr']=}")
        except Exception as e:
            print(e)
            metrics["auc_fdr_tpr"] = -1
        # metrics["auc_new"] = auc_new

        if self.save_output:
            # path_output = os.path.join(os.getcwd(), "output_detection.pt")
            torch.save(self.to_save, self.path_output)
            print(f"Output saved to {self.path_output}")
            self.print_path(self.path_output)
            self.to_save = defaultdict(list)

        self.all_data = defaultdict(list)

        if False:
            print(f"{thresholds=}")
            # thresholds = thresholds[1:-1]
            # tpr = tpr[1:-1]
            # plt.plot(thresholds, tpr)
            # plt.scatter(fdr, tpr)
            # plt.plot(fdr, tpr)
            # plt.plot(precision, recall)
            plt.plot(recall, precision)
            # plt.scatter(fdr, tpr)
            plt.scatter(recall, precision)
            plt.show()

        # metrics["snr"] = snr
        # breakpoint()
        return {"metrics": metrics, "snr": np.stack(snr)}
        # return {"metrics": metrics}

    def print_path(self, path):
        sep = "run/"
        rel_path = path.split(sep)[-1]
        rel_path = os.path.join("data", sep, rel_path)
        print(f"'{self.run_name}': '{rel_path}',")


# class AUC:
    # def __init__(self, kernel_size, circle=False, size=256):
        # self.local_masker = LocalMasker(
            # kernel_size=1 + 2 * kernel_size // 2, circle=circle, size=size
        # )
        # self.reset()

    # def reset(self):
        # self.all_values = []
        # self.all_cat = []

    # def add(self, snr, mask_valid, mask_cat):
        # H, W = snr.shape
        # H, W = mask_valid.shape
        # H, W = mask_cat.shape
        # mask_local = (
            # self.local_masker(torch.tensor(snr).view(1, 1, H, W).to(device))[
                # 0, 0
            # ]
            # .cpu()
            # .numpy()
        # )
        # mask_local = mask_valid.astype(bool) & mask_local.astype(bool)

        # mask_cat = mask_cat.astype(bool)
        # self.all_values.append(snr[mask_local])
        # self.all_cat.append(mask_cat[mask_local])

    # # def process(self, all_snr, all_mask):
    # def process(self):
        # all_values = np.concatenate(self.all_values)
        # all_cat = np.concatenate(self.all_cat)

        # val_1 = all_values[all_cat]
        # val_0 = all_values[~all_cat]
        # n0 = len(val_0)
        # n1 = len(val_1)

        # diff = val_0.reshape(-1, 1) - val_1.reshape(1, -1)
        # error = diff > 0
        # auc = 1 - error.sum()/ (n0 * n1)

        # return auc




def check_metrics():
    path = "data/run/eval_detection/paired/2023-10-11_00-11-46/sampling_direct_iteration_t64_st8_s32_ss8/output_detection.pt"
    device = torch.device("cpu")
    data = torch.load(path, map_location=device)
    snr = data["snr"][0][0]
    labels_gt = data["labels_gt"][0]
    # markers, _ = ndi.label(labels_gt)
    # n_gt = len(np.unique(markers)) - 1
    labels_real = np.zeros_like(labels_gt)
    tp, fp, fn = get_result_thresh(
        detection_map=snr,
        labels_gt=labels_gt,
        labels_real=labels_real,
        threshold=1,
        radius=9,
    )


if __name__ == "__main__":
    check_metrics()
