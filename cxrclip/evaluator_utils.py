
import logging
import numpy as np
from sklearn import metrics
import torch
from tqdm import tqdm
from statistics import mean
from collections import Counter
import cv2
import os
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore
import pickle
from cxrclip.prompt import constants
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)

def get_class_list(dataset_name):
    class_list = None
    if hasattr(constants, dataset_name.upper()): # NOTE: when you add new dataset, specify the class name in the constants file.
        class_list = getattr(constants, dataset_name.upper())
    elif dataset_name == 'padchest_gr':
        with open("/cluster/projects/mcintoshgroup/CXR-CLIP/preprossed_dataset_csv/padchest_gr/padchest_gr_official_boxes_unique_phrases.pkl", "rb") as f:
            class_list = pickle.load(f)
    return class_list

def mask2rle(mask):
    """Accurate inverse of your rle2mask function"""
    # 1. We must flatten in Fortran order ('F') to match the .T logic
    pixels = mask.flatten(order='F')
    
    # 2. Find where the values change
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    
    # 3. Calculate absolute starts and lengths
    abs_starts = runs[0::2]
    lengths = runs[1::2] - abs_starts
    
    # 4. Calculate RELATIVE starts (since your function uses current_position += start)
    rel_starts = np.copy(abs_starts)
    # The first gap is just the first start index.
    # Subsequent gaps are: current_start - previous_end
    if len(abs_starts) > 1:
        prev_ends = abs_starts[:-1] + lengths[:-1]
        rel_starts[1:] -= prev_ends

    # 5. Interleave and stringify
    res = np.zeros(len(rel_starts) * 2, dtype=int)
    res[0::2] = rel_starts
    res[1::2] = lengths
    
    return ' '.join(map(str, res))

def rle2mask(rle, width, height):
    """Run length encoding to segmentation mask"""
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 1
        current_position += lengths[index]
    return mask.reshape(width, height).T

def gather_seg_statistics(
        attention_maps, 
        image_paths, 
        masks, 
        label_names, 
        class_list, 
        input_res=None,
        save_dir=None
    ):
    """
    check the segmentation_utils.py file.
    """
    assert len(masks) == len(label_names) and len(label_names) == len(image_paths), "size does not match."
    mask_labels = list(zip(masks, label_names))
    B = len(masks)
    Q = len(class_list)

    masks_tensor = [[[] for _ in range(Q)] for _ in range(B)]
    for i, (m, names) in enumerate(mask_labels):
        for bbox, disease_name in zip(m, names):
            if disease_name in class_list: # exact match case insensitive.
                q = class_list.index(disease_name)
                masks_tensor[i][q].append(bbox)  # APPEND, don't overwrite
    
    if isinstance(attention_maps, np.ndarray):
        attention_maps=torch.from_numpy(attention_maps).clone()
    assert len(masks) == attention_maps.shape[0], 'number of the attention maps should matches to number of ground truth masks'

    # min-max normalize the attention map so that it can be properly thresholded for the dice score and etc.
    min_val = attention_maps.amin(dim=-1, keepdim=True)
    max_val = attention_maps.amax(dim=-1, keepdim=True)
    attention_maps = (attention_maps - min_val) / (max_val - min_val + 1e-8)

    B, Q, T = attention_maps.shape
    assert T > 1
    assert T in {1370, 1369, 257, 256, 325, 324, 197, 196, 49}, \
        "Unexpected token count. Ensure attn includes/excludes CLS consistently."

    # ---- remove CLS if present ----
    if T in {1370, 257, 325, 197}:
        spatial = attention_maps[:, :, 1:]   # [B, Q, HW]
    else:
        spatial = attention_maps             # [B, Q, HW]

    if input_res is not None:
        S = input_res
    elif T == 1370 or T == 1369: S = 518
    elif T == 325 or T == 324: S = 256
    elif T == 257 or T == 256: S = 224
    else: S = 224

    HW = spatial.shape[-1]
    Hg = Wg = int(HW ** 0.5)
    assert Hg * Wg == HW, "Token count (without CLS) must form a square grid."

    # [B, Q, Hg, Wg], the spatial attention map
    spatial = spatial.reshape(B, Q, Hg, Wg)

    # ---- load original size images ----
    img_sizes = []
    for path in tqdm(image_paths, desc="Loading image dimensions"):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"cv2 cannot read image: {path}")
        H0, W0 = img.shape[:2]
        img_sizes.append((H0, W0))

    global_results = {}
    for q in range(Q):
        positive_probs, positive_masks = [], []
        unscaled_heatmap = []
        valid_image_paths =[]
        for i in range(B):
            mask = masks_tensor[i][q]

            # skip the non-annotated ones.
            if len(mask) == 0:
                continue
            assert len(mask) == 1, 'something wrong'
            mask = mask[0]
            valid_image_paths.append(image_paths[i]) # <--- Add this
            H0, W0 = img_sizes[i]
            heat = spatial[i, q].unsqueeze(0).unsqueeze(0)  # [1,1,Hg,Wg]

            if H0 == W0:
                # ---- upsample patch map to ORIGINAL image size, assume BlipProcessor ----
                heat = F.interpolate(
                    heat, size=(H0, W0),
                    mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)  # [H0, W0]
                prob = torch.sigmoid(heat)
                prob = prob.detach().cpu()
            else:
                scale = S / float(min(H0, W0))
                Hr = int(round(H0 * scale))
                Wr = int(round(W0 * scale))

                # CenterCrop offsets on resized image
                top  = max((Hr - S) // 2, 0)
                left = max((Wr - S) // 2, 0)

                # 1) upsample to crop space (S,S)
                heat = F.interpolate(heat, size=(S, S), mode="bilinear", align_corners=False)[0, 0]
                prob_crop = torch.sigmoid(heat)  # [S,S]

                # 2) paste crop back into resized canvas (Hr,Wr)
                canvas = torch.zeros((Hr, Wr), device=prob_crop.device, dtype=prob_crop.dtype)
                canvas[top:top+S, left:left+S] = prob_crop

                # 3) resize the prob_crop version -> original size to compare with GT mask.
                prob_orig = F.interpolate(
                    canvas.unsqueeze(0).unsqueeze(0),
                    size=(H0, W0),
                    mode="bilinear",
                    align_corners=False
                )[0, 0]  # [H0,W0]
                prob = prob_orig.detach().cpu()

            assert mask.sum() > 0
            unscaled_heatmap.append(heat)
            positive_probs.append(prob)
            positive_masks.append(mask)

        class_result = {}
        
        if len(positive_probs) == 0:
            print(f'class {class_list[q]} has no annotation, skipping')
            continue

        # dice score
        ref_shape = positive_probs[0].shape
        same_shape = all(t.shape == ref_shape for t in positive_probs)
        positive_probs_list = positive_probs
        positive_masks_list = positive_masks
        if same_shape:
            positive_probs = torch.stack(positive_probs).unsqueeze(1).cpu()
            positive_masks = torch.stack(positive_masks).unsqueeze(1).cpu()

        best_dice = 0.0
        for t in tqdm(
            np.arange(0, 1.01, 0.01), 
            desc=f"Finding optimal threshold for {class_list[q]}"
        ):
            if same_shape:
                # batch processing
                dice = DiceScore(
                    num_classes=2,
                    include_background=False,
                    average='macro',
                    aggregation_level='samplewise'
                )
                cur_dice = dice((positive_probs > t).long(), positive_masks)
                if cur_dice > best_dice:
                    best_dice = cur_dice
                    best_threshold = t
            else:
                # one at a time - SLOW but necessary for fallback option.
                dices = []
                for prob, mask in zip(positive_probs, positive_masks):
                    pred = (prob > t).long().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                    gt = mask.long().unsqueeze(0).unsqueeze(0)

                    dice = DiceScore(num_classes=1)
                    cur = dice(pred, gt)
                    dices.append(cur)
                mean_dice = torch.stack(dices).mean()
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_threshold = t

        dice_score = best_dice.item()

        print(f"dice score (positive only) for {class_list[q]}: {dice_score}")
        print(f"best threshold: {best_threshold}")
        class_result["zeroshot_dice"] = dice_score
        class_result["best_threshold"] = best_threshold
        global_results[class_list[q]] = class_result

    return global_results


def define_fname_by_path(img_path):

    # Normalize path (handles Windows/Linux)
    img_path = os.path.normpath(img_path)
    parts = img_path.split(os.sep)
    
    if 'chexlocalize' in parts:
        idx = parts.index('chexlocalize')
        
        # Exclude filename (last element)
        relevant_parts = parts[idx:-1]
        # Add filename without extension
        file_stem = os.path.splitext(parts[-1])[0]
        fname = "_".join(relevant_parts + [file_stem])
    else:
        fname = '.'.join(os.path.basename(img_path).split('.')[:-1])
    return fname

def compute_specificity(negative_probs, threshold):
    """
    Calculate image-level specificity.

    Specificity is the proportion of true negatives (images correctly identified as negative)
    out of the total number of negatives.

    Args:
        negative_probs (torch.Tensor): Tensor of predicted probabilities for negative images.
        threshold (float): Threshold to classify the predictions.

    Returns:
        float: Specificity value.
    """
    # Check if all pixels in the negative images are below the threshold
    true_negatives = (
        (negative_probs.squeeze(1) > threshold).long().sum(-1).sum(-1) == 0
    ).sum()

    # Calculate specificity as the ratio of true negatives to the total number of negative images
    spec = (true_negatives / len(negative_probs)).item()
    return spec

def get_grounding_type(dataset_name):
    if dataset_name in {'chexlocalize'}:
        g_type = 'contour'
    elif dataset_name in {'ms_cxr_test', 'ms_cxr_test_v2', 'padchest_gr'}:
        g_type = 'refer_grounding'
    elif dataset_name in {'siim_pneumothorax_seg', 'rsna_pneumonia_seg', 'qatacovid_seg', 'chexlocalize_seg'}:
        g_type = 'mask'
    else:
        g_type = 'pointing_game'
    return g_type

def gather_pointinggame_statistics(
        class_list, 
        attention_maps, 
        bbox_labels, 
        label_names, 
        image_paths, 
        grounding_type='pointing_game',
        input_res=None,
        visual_save_dir=None,
        visual_name_suffix='',
        draw_box_on_overlay=True
    ):
    assert grounding_type in ['pointing_game', 'contour', 'refer_grounding', 'mask']
    """
    pointinggame_predictions is shape [number of images, number of class, prediction of each class]
    """
    assert len(bbox_labels) == len(label_names) and len(label_names) == len(image_paths), "size does not match."
    bbox_labels = list(zip(bbox_labels, label_names))
    B = len(bbox_labels)
    Q = len(class_list)

    bbox_tensor = [[[] for _ in range(Q)] for _ in range(B)]
    for i, (bboxes, names) in enumerate(bbox_labels):
        for bbox, disease_name in zip(bboxes, names):
            if disease_name in class_list: # exact match case insensitive.
                q = class_list.index(disease_name)
                bbox_tensor[i][q].append(list(bbox))  # APPEND, don't overwrite
    
    if isinstance(attention_maps, np.ndarray):
        attention_maps = torch.from_numpy(attention_maps).clone()
    else:
        attention_maps = attention_maps.clone()

    if grounding_type in {'pointing_game', 'contour', 'mask'}:
        results = pointing_game(
            attention_maps, 
            bbox_tensor, 
            image_paths, 
            label_names, 
            class_list, 
            grounding_type=grounding_type,
            input_res=input_res,
            save_dir=visual_save_dir,
            visual_fname_suffix=visual_name_suffix,
            draw_box_on_overlay=draw_box_on_overlay
        )
    else:
        results = pointing_game_flat(
            attention_maps, 
            bbox_tensor, 
            image_paths, 
            class_list,
            input_res=input_res,
            save_dir=visual_save_dir,
            visual_fname_suffix=visual_name_suffix,
            draw_box_on_overlay=draw_box_on_overlay
            )
    return results

def pointing_game_flat(
        attn, 
        bbox, 
        image_paths, 
        class_list, 
        input_res=None,
        save_dir=None,
        visual_fname_suffix='',
        draw_box_on_overlay=True
    ):
    """
    Modified Pointing Game:
    Returns a flat list of predictions (1 for Hit, 0 for Miss) for every valid 
    (image, class) pair where a ground truth box exists.

    Returns:
        hit_results: List[int] containing 0s and 1s.
    """
    hit_results = []
    
    B, Q, T = attn.shape
    assert T > 1
    
    # ---- remove CLS if present ----
    if T in {1370, 257, 325, 197}:
        spatial = attn[:, :, 1:]   # [B, Q, HW]
    else:
        spatial = attn             # [B, Q, HW]

    if input_res is not None:
        S = input_res
    elif T == 1370 or T == 1369: S = 518
    elif T == 325 or T == 324: S = 256
    elif T == 257 or T == 256: S = 224
    else: S = 224

    HW = spatial.shape[-1]
    Hg = Wg = int(HW ** 0.5)
    
    # [B, Q, Hg, Wg]
    spatial = spatial.reshape(B, Q, Hg, Wg)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # ---- load original sizes once ----
    # (Optimization Note: Ideally pass these in instead of reading from disk)
    img_data = []
    for path in tqdm(image_paths, desc="Loading image dimensions"):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"cv2 cannot read image: {path}")
        img_data.append(img.shape[:2])

    def _boxes_for(i, q):
        """Return list of boxes for sample (i,q) or empty list."""
        b = bbox[i][q]
        if b is None or len(b) == 0: return []
        if isinstance(b, (list, tuple)) and len(b) > 0 and isinstance(b[0], (list, tuple, np.ndarray)):
            return b
        if isinstance(b, (list, tuple, np.ndarray)) and len(b) == 4:
            return [b]
        return []

    def _point_in_any_box(box_list, x, y):
        """mainly generalize to padchest_gr and ms_cxr"""
        # assert len(box_list) == 1
        for box in box_list:
            x_min, y_min, x_max, y_max = box
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    # Iterate over all Classes (Q) and Images (B)
    for q in range(Q):
        cls_name = class_list[q]

        for i in range(B):
            box_list = _boxes_for(i, q)
            
            # Skip samples where this class is not in the ground truth
            if len(box_list) == 0:
                continue 
            
            H0, W0 = img_data[i]

            scale = S / min(H0, W0)
            Hr, Wr = int(round(H0 * scale)), int(round(W0 * scale))

            # center crop offsets in the resized image
            top  = max((Hr - S) // 2, 0)
            left = max((Wr - S) // 2, 0)

            # 1) upsample patch heatmap to crop size (S,S), NOT (H0,W0)
            heat = spatial[i, q].unsqueeze(0).unsqueeze(0)  # 1x1xhxw
            heat = F.interpolate(heat, size=(S, S), mode="bilinear", align_corners=False)[0, 0]

            # 2) argmax in crop coords
            idx = int(torch.argmax(heat).item())
            y_c = idx // S
            x_c = idx % S

            # 3) crop coords -> resized coords
            x_r = x_c + left
            y_r = y_c + top

            # 4) resized coords -> original coords
            x_peak = int(round(x_r / scale))
            y_peak = int(round(y_r / scale))

            # Check Hit or Miss
            hit = _point_in_any_box(box_list, x_peak, y_peak)
            hit_results.append(int(hit))

    return {
        'refer_grounding': hit_results
    } 

def pointing_game(
        attn, 
        bbox, 
        image_paths, 
        label_names, 
        class_list, 
        grounding_type,
        input_res=None,
        save_dir=None,
        visual_fname_suffix='',
        draw_box_on_overlay=True
    ):
    """
    Pointing Game metric in the same *style* as chestXDet10_eval_grounding:
    - For each (image, query), get a spatial score map over patches.
    - Convert patch grid -> original image resolution (H0, W0) by bilinear upsampling.
    - Take argmax point in ORIGINAL coords.
    - Hit if point is inside ANY GT box for that (image, query).
    - Skip samples where no GT box exists for that (image, query).
    
    Args:
        attn:  [B, Q, T] attention/similarity over tokens (may include CLS)
        bbox:  nested boxes. bbox[i][q] should be:
            - [] or None if absent, OR
            - list of boxes [[xmin,ymin,xmax,ymax], ...] in ORIGINAL image coords
        image_paths: list length B, used to read original sizes
        class_list: list length Q

    Returns:
        results: dict[class_name]['pointing_game'] = hit-rate over images where GT present
    """
    results = {}
    B, Q, T = attn.shape
    assert T > 1
    assert T in {1370, 1369, 257, 256, 325, 324, 197, 196, 49}, \
        "Unexpected token count. Ensure attn includes/excludes CLS consistently."

    # ---- remove CLS if present ----
    if T in {1370, 257, 325, 197}:
        spatial = attn[:, :, 1:]   # [B, Q, HW]
    else:
        spatial = attn             # [B, Q, HW]
    
    if input_res is not None:
        S = input_res
    elif T == 1370 or T == 1369: S = 518
    elif T == 325 or T == 324: S = 256
    elif T == 257 or T == 256: S = 224
    else: S = 224
    # NOTE: for biovil-t 196 corresponds to 448 input size

    HW = spatial.shape[-1]
    Hg = Wg = int(HW ** 0.5)
    assert Hg * Wg == HW, "Token count (without CLS) must form a square grid."

    # [B, Q, Hg, Wg]
    spatial = spatial.reshape(B, Q, Hg, Wg)

    # ---- load original sizes once ----
    img_data = []
    for path in tqdm(image_paths, desc="Loading image dimensions"):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"cv2 cannot read image: {path}")
        img_data.append(img)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    def _boxes_for(i, q):
        """Return list of boxes for sample (i,q) or empty list."""
        b = bbox[i][q]
        if b is None:
            return []
        # already list-of-boxes
        if isinstance(b, (list, tuple)) and len(b) > 0 and isinstance(b[0], (list, tuple, np.ndarray)):
            return b
        # single box
        if isinstance(b, (list, tuple, np.ndarray)) and len(b) == 4:
            return [b]
        return []

    def _point_in_any_box(box_list, x, y):
        for box in box_list:
            x_min, y_min, x_max, y_max = box
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def _pointing_game_hit_by_mask(mask, x, y):
        """
        mainly for siim and other datasets that provides the segmentation masks only
        mainly for pointing game for segmentation mask instead of bounding box.
        heat: (H, W) float heatmap
        mask: (H, W) binary mask for the class
        """
        y, x = np.unravel_index(np.argmax(heat), heat.shape)
        hit = bool(mask[y, x] > 0)
        return hit
    
    def _hit_by_contours(contours_xy, x, y):
        """
        mainly for chestXlocalize
        point_xy: (x, y) in image pixel coordinates
        contours_xy: list of polygons, each polygon is [(x,y), (x,y), ...], essentially a list of list.
        returns: True if inside ANY polygon
        """
        pt = (float(x), float(y))
        for poly in contours_xy:
            poly_np = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)  # (N,1,2)
            inside = cv2.pointPolygonTest(poly_np, pt, measureDist=False)  # 1 inside, 0 on edge, -1 outside
            if inside >= 0:
                return True
        return False

    assert grounding_type in ['pointing_game', 'contour', 'mask']
    if grounding_type == 'pointing_game':
        hit_func = _point_in_any_box
    elif grounding_type == 'contour':
        hit_func = _hit_by_contours
    elif grounding_type == 'mask':
        hit_func = _pointing_game_hit_by_mask

    for q in range(Q):
        hits_q, count_q = 0, 0
        cls_name = class_list[q]

        for i in range(B):
            box_list = _boxes_for(i, q)
            if len(box_list) == 0:
                continue  # skip if no GT for this (image, query)

            orig_img = img_data[i].copy()
            H0, W0 = orig_img.shape[:2]
            # assert H0 == W0,\
            # 'The following implementation assume the original image is in squared size. Otherwise it need to handle the Resize() operation part.'
            
            heat = spatial[i, q].unsqueeze(0).unsqueeze(0)  # [1,1,Hg,Wg]
            if H0 == W0:
                # ---- upsample patch map to ORIGINAL image size ----
                heat = F.interpolate(
                    heat, size=(H0, W0),
                    mode="bilinear", 
                    align_corners=False
                ).squeeze(0).squeeze(0)  # [H0, W0]
            else:
                # scale so that min(H0, W0) -> S
                scale = S / float(min(H0, W0))

                # resized size
                Hr = int(round(H0 * scale))
                Wr = int(round(W0 * scale))

                # center crop offsets in resized image
                top  = max((Hr - S) // 2, 0)
                left = max((Wr - S) // 2, 0)

                # 1) upsample patch heatmap to crop size (S,S), NOT (H0,W0)
                heat = F.interpolate(heat, size=(S, S), mode="bilinear", align_corners=False)[0, 0]

            # argmax in ORIGINAL coords
            idx = int(torch.argmax(heat).item())
            if H0 == W0:
                y_peak = idx // W0
                x_peak = idx % W0
            else:
                y_c = idx // S
                x_c = idx % S

                # 3) crop coords -> resized coords
                x_r = x_c + left
                y_r = y_c + top
                # 4) resized coords -> original coords
                x_peak = int(round(x_r / scale))
                y_peak = int(round(y_r / scale))

            # route to different method based on the dataset: bounding box, contour, and segmentation
            hit = hit_func(box_list, x_peak, y_peak)
            hits_q += int(hit)
            count_q += 1

        if count_q > 0:
            results.setdefault(cls_name, {})
            results[cls_name]["pointing_game"] = hits_q / count_q

    return results

def text_guided_similarities(image_embeddings, image_patch_embeddings, text_embeddings):

    similarities = []
    for i, text in enumerate(text_embeddings):
        enriched_image_embedddings_for_i = flair_attention_numpy(
            text[None, None, :], 
            np.concatenate([image_embeddings[:, None, :], image_patch_embeddings], axis=1), # combine the cls with patch tokens
            None
        ) # => in shape [3000x512]

        similarity_scores = enriched_image_embedddings_for_i @ text.reshape(-1, 1)
        similarities.append(similarity_scores)

    similarities = np.concatenate(similarities, axis=-1)
    return similarities

def multilabel_classification(preds: np.ndarray, labels: np.ndarray, class_list: list):
    log.info("evaluate multi-label classification")

    result = {}
    for idx, class_name in enumerate(class_list):
        result[class_name] = {}
        fpr, tpr, thresholds = metrics.roc_curve(labels[:, idx], preds[:, idx])
        result[class_name]["AUROC"] = metrics.auc(fpr, tpr)
        result[class_name]["PR_AUROC"] = metrics.average_precision_score(labels[:, idx], preds[:, idx])

        result[class_name]["Accuracy"] = metrics.accuracy_score(labels[:, idx], preds[:, idx] > 0.5)
        result[class_name]["F1"] = metrics.f1_score(labels[:, idx], preds[:, idx] > 0.5)

    return classification_score(result)


def pointing_game_score(result: dict, _print=True):
    pg = np.mean([value["pointing_game"] for value in result.values()])
    result["Pointing_Game(Avg)"] = pg
    if _print:
        s = "\n".join(f"{k}: {v}" for k, v in result.items())
        log.info(s)
    return result

def refer_grounding_score(result: dict, _print=True):
    pg = sum(result['refer_grounding']) / len(result['refer_grounding'])
    result["zeroshot_refer_grounding(Acc)"] = pg
    result['refer_grounding'] = pg
    if _print:
        s = "\n".join(f"{k}: {v}" for k, v in result.items())
        log.info(s)
    return result

def seg_score(result: dict, _print=True):
    pg = np.mean([value["zeroshot_dice"] for value in result.values()])
    result["zeroshot_segmentation(Avg)"] = pg
    if _print:
        s = "\n".join(f"{k}: {v}" for k, v in result.items())
        log.info(s)
    return result

def classification_score(result: dict, class_counts: dict = {}, _print=True):

    # macro average
    auroc = np.mean([value["AUROC"] for value in result.values()])
    acc = np.mean([value["Accuracy"] for value in result.values()])
    pr_auc = np.mean([value["PR_AUROC"] for value in result.values()])

    result["AUROC(Avg)"] = auroc
    result["PR_AUROC(Avg)"] = pr_auc
    result["Accuracy(Avg)"] = acc
    # ---------- weighted average ----------
    if class_counts:
        weights = np.array([class_counts[k] for k in result.keys() if k in class_counts])
        weights = weights / weights.sum()

        def weighted_metric(metric_name):
            vals = []
            wts = []
            for k, v in result.items():
                if k in class_counts and metric_name in v:
                    vals.append(v[metric_name])
                    wts.append(class_counts[k])
            if not vals:
                return None
            wts = np.array(wts) / np.sum(wts)
            return np.sum(np.array(vals) * wts)

        w_auroc = weighted_metric("AUROC")
        w_pr_auc = weighted_metric("PR_AUROC")
        w_acc = weighted_metric("Accuracy")


        if w_auroc is not None:
            result["AUROC(Weighted)"] = w_auroc
        if w_pr_auc is not None:
            result["PR_AUROC(Weighted)"] = w_pr_auc
        if w_acc is not None:
            result["Accuracy(Weighted)"] = w_acc

    if _print:
        s = "\n".join(f"{k}: {v}" for k, v in result.items())
        log.info(s)

    return result

def multiclass_classification(preds: np.ndarray, labels: np.ndarray, class_list: list):
    log.info("evaluate multi-class classification")
    preds_args = np.argmax(preds, axis=1)

    class_dict = {class_name: {"total_num": 0, "correct_num": 0} for class_name in class_list}
    for idx, class_name in enumerate(class_list):
        class_dict[class_name]["total_num"] = labels[:, idx].sum()
        class_dict[class_name]["correct_num"] = (labels[:, idx] * (preds_args == idx)).sum()

    total_num = len(labels)
    correct_num = sum([v["correct_num"] for v in class_dict.values()])

    result = {k: v["correct_num"] / v["total_num"] for k, v in class_dict.items()}
    result["Accuracy(Macro)"] = np.mean(list(result.values()))
    result["Accuracy(Micro)"] = correct_num / total_num  # same with macro due to same total_num
    s = " / ".join([f"{c}: {v:.3f}" for c, v in result.items()])
    log.info(s)

    return result

def chunked_colbert_sim(
    query_local_features: torch.Tensor, # [b, n, d], image or text local and cls features
    values_local_features: torch.Tensor, # [B, m, d], image or text local and cls features
    attention_mask: torch.Tensor, # [b, n] or [B, m], attention mask on either query features or values features
    similarity_type: str,
    mask_on_query: bool = True, 
    reduce_type: str = 'average',
    batch_size: int = 2) -> torch.Tensor:

    with torch.no_grad():
        b, n, d = query_local_features.shape
        B, m, d2 = values_local_features.shape

        # sanity checks.
        assert d == d2, "Embedding dimensions must match"

        b, n, d = query_local_features.shape
        B, m, _ = values_local_features.shape
        colbert_scores = []
        # for i in range(0, b, batch_size):
        for i in tqdm(range(0, b, batch_size), desc="Query Batches"):
            q_end = min(i + batch_size, b)
            q_chunk = query_local_features[i:q_end]  # (q_bs, n, d)

            sims_cols = []
            for j in range(0, B, batch_size):
                v_end = min(j + batch_size, B)
                v_chunk = values_local_features[j:v_end]  # (v_bs, m, d)

                # (q_bs, n, d) x (v_bs, m, d) -> (q_bs, n, v_bs, m)
                sim = torch.einsum("bnd,Bmd->bnBm", q_chunk, v_chunk)
                sims_cols.append(sim)  # along B

            # similarity scores between batch-wise query and value
            sims = torch.cat(sims_cols, dim=2)  # concat along B

            # Apply mask BEFORE max operation on the value side.
            if not mask_on_query and attention_mask is not None: # mask on value=text
                # assert attention_mask.shape == (b, m)
                mask = attention_mask[None, None, :, :]  # => [1, 1, B, m]
                sims = sims.masked_fill(mask == 0, -9999.0) # prevent the padding features has the max similarity to the query tokens

            if similarity_type == 'COLBERT_MAX':
                sims, max_indices = sims.max(dim=3) # MaxSim operation on the value tokens (m) => [b, n, B]
            elif similarity_type == 'COLBERT_SOFTMAX':
                softmax_sim = torch.softmax(sims, dim=3) # [b, n, B, m], each of the n query token has m value tokens to softmax
                sims = (softmax_sim * sims).sum(3) # [b, n, B], each of the n is already weighted sum of the m values
                
            # Mask max_sim for reduction
            if mask_on_query:
                sims = sims * attention_mask[i:q_end, :, None].float()  # [b, n, B]
                valid_query_token_counts = attention_mask[i:q_end].sum(dim=1)  # [b] number of word tokens
            else:
                # keep the max_sim as it is as m2 are all valid feature embeddings, not padding features or cls features
                valid_query_token_counts = torch.ones(q_chunk.shape[0], device=attention_mask.device) * query_local_features.shape[1]  # all valid in m1 => [B]

            # Sum or average over the query tokens (never the value tokens): [b, b]
            if reduce_type == 'sum':
                scores = sims.sum(dim=1)  # [b, n, B] => [b, B]
            elif reduce_type == 'average':
                scores = sims.sum(dim=1) / valid_query_token_counts.unsqueeze(-1) # [b, n, B] => [b, B] / [b, 1]
            else:
                raise ValueError(f"Unknown reduce_type: {reduce_type}")

            colbert_scores.append(scores.detach().cpu().numpy())

    # the colbert similarity scores.
    return np.concatenate(colbert_scores, axis=0)

def flair_attention_numpy(query_feats, value_feats, value_masks):
    """
    numpy version of the function in cxrclip_plusplus.py

    query_feats: [B, 1, D]
    value_feats: [B, P, D]
    value_masks: [B, P, 1]
    return results : [B, 1, D], each D is the weighted summation of the attended features of the value based on a single query features
    """
    if value_masks is not None:
        assert value_masks.shape[1] == value_feats.shape[1], "number of tokens should be the same for word tokens and mask tokens"

    # Cosine similarity: dot product over last dim -> [B, P]
    attn_scores = np.sum(query_feats * value_feats, axis=-1)

    if value_masks is not None:
        mask = value_masks.squeeze(-1)  # [B, P]
        attn_scores = np.where(mask == 1, attn_scores, -np.inf)  # mask out with -inf

    attn_exp = np.exp(attn_scores)
    
    if value_masks is not None:
        attn_exp = attn_exp * mask  # zero out invalid positions

    attn_weights = attn_exp / np.sum(attn_exp, axis=-1, keepdims=True)  # [B, P]

    # Weighted sum: [B, P, 1] * [B, P, D] = [B, P, D], then sum over P -> [B, D]
    weighted = attn_weights[..., np.newaxis] * value_feats
    result = np.sum(weighted, axis=1, keepdims=True)  # [B, 1, D]

    return result.squeeze()

def disease_list_mapping(names):    
    return [disease_name_mapping(name) for name in names]

def disease_name_mapping(name):
    if name.lower() == 'ILD'.lower() or 'interstitial lung disease' in name.lower():
        return 'interstitial lung disease'
    if name.lower() == 'Enlarged PA'.lower():
        return 'enlarged pulmonary artery'
    if name.lower() == 'COPD'.lower():
        return 'chronic obstructive pulmonary disease'
    if name.lower() == 'Pleural_Thickening'.lower():
        return 'pleural thickening'
    if name.lower() == 'No Finding'.lower() or name.lower() == 'Normal'.lower() or name.lower() == 'No Findings'.lower() or 'no ' in name.lower():
        return 'no findings'
    if name.lower() == 'Fracture'.lower():
        return 'rib fracture'
    if name.lower() == 'Nodule'.lower():
        return 'lung nodule'
    if name.lower() == 'Mass'.lower():
        return 'lung mass'
    if name.lower() == 'Effusion'.lower():
        return 'pleural effusion'
    if name.lower() == 'Fibrosis'.lower():
        return 'pulmonary fibrosis'
    if name.lower() == 'cpam'.lower():
        return 'congenital pulmonary airway malformation'
    if name.lower() == 'COPD'.lower():
        return 'chronic obstructive pulmonary disease'
    if name.lower() == 'tb':
        return 'tuberculosis'
    if name.lower() == 'fracture old' or name.lower() == 'fracture_old':
        return 'fracture'
    if name.lower() == 'nodule/mass':
        return 'nodule or mass'
    covid_des = 'ground-glass opacities, consolidation, pleural thickening commonly appear in infection'
    if name.lower() == 'covid'.lower():
        return covid_des
    return name.lower()

def evals_to_csv_rows(evals: dict, dataset_name: str) -> list:
    """
    Flatten per-checkpoint evaluation results into a list of row dicts for CSV export.

    Parameters
    ----------
    evals : dict
        ``{ ckpt_path: { task_name: result_dict } }`` as returned by
        ``evaluate_clip_offline`` for one dataset.
    dataset_name : str
        Name of the dataset being evaluated (e.g. ``"chest14"``).

    Returns
    -------
    list of dict, each with keys:
        checkpoint       – full checkpoint path
        checkpoint_name  – filename stem (e.g. ``"model-5"``)
        epoch            – integer epoch parsed from checkpoint_name; None if unparseable
        dataset_name     – dataset being evaluated
        disease          – disease/class name; ``"ALL"`` for aggregate rows
        task_name        – one of the four supported task keys (see below)
        metric_name      – ``AUROC`` | ``pointing_game`` | ``zeroshot_dice`` |
                           ``refer_grounding_acc``
        metric_value     – float
        is_aggregate     – ``True`` for the dataset-level macro-average row (disease=``"ALL"``),
                           ``False`` for per-disease rows

    Supported tasks and the single metric extracted per task
    --------------------------------------------------------
    zeroshot_binary          -> AUROC   (per-disease + macro-avg aggregate)
    zeroshot_pointing_game   -> pointing_game  (per-disease + macro-avg aggregate)
    zeroshot_seg             -> zeroshot_dice  (per-disease + macro-avg aggregate)
    zeroshot_refer_grounding -> refer_grounding_acc  (aggregate only; no per-disease breakdown)

    Per-disease and aggregate rows share the same ``metric_name`` so that a
    filter like ``df[df.metric_name == "AUROC"]`` returns both granularities,
    distinguished by ``is_aggregate``.
    """
    import os

    # Maps internal result-dict task keys to human-readable task_name values in the CSV.
    _TASK_LABEL = {
        'zeroshot_binary':          'classification',
        'zeroshot_pointing_game':   'grounding',
        'zeroshot_seg':             'segmentation',
        'zeroshot_refer_grounding': 'phrase_grounding',
    }

    # (internal_task_key, per_class_metric_key, aggregate_result_dict_key)
    _TASK_SPEC = [
        ('zeroshot_binary',        'AUROC',         'AUROC(Avg)'),
        ('zeroshot_pointing_game', 'pointing_game', 'Pointing_Game(Avg)'),
        ('zeroshot_seg',           'zeroshot_dice', 'zeroshot_segmentation(Avg)'),
    ]

    rows = []
    for ckpt_path, task_results in evals.items():
        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        try:
            epoch = int(ckpt_stem.rsplit('-', 1)[-1])
        except (ValueError, IndexError):
            epoch = None

        base = dict(
            checkpoint=ckpt_path,
            checkpoint_name=ckpt_stem,
            epoch=epoch,
            dataset_name=dataset_name,
        )

        def _row(disease, task_label, metric, value, is_agg, _base=base):
            return {**_base, 'disease': disease, 'task_name': task_label,
                    'metric_name': metric, 'metric_value': float(value),
                    'is_average': is_agg}

        # --- tasks with per-disease breakdown ---
        for internal_key, per_class_key, agg_key in _TASK_SPEC:
            result = task_results.get(internal_key)
            if result is None:
                continue
            label = _TASK_LABEL[internal_key]
            for key, val in result.items():
                if isinstance(val, dict):
                    if per_class_key in val:
                        rows.append(_row(key, label, per_class_key, val[per_class_key], False))
                elif key == agg_key and isinstance(val, (int, float)):
                    rows.append(_row('ALL', label, per_class_key, val, True))

        # --- phrase grounding: aggregate only ---
        rg = task_results.get('zeroshot_refer_grounding')
        if rg is not None and 'zeroshot_refer_grounding(Acc)' in rg:
            rows.append(_row('ALL', _TASK_LABEL['zeroshot_refer_grounding'], 'refer_grounding_acc',
                             rg['zeroshot_refer_grounding(Acc)'], True))

    return rows