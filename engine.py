import math
import os
import sys
from typing import Iterable
import venv
import numpy as np
import copy
import itertools
import pickle
import ivtmetrics
import torch
from datasets.hico_text_label import hico_text_label
from util.topk import top_k
from models.matcher import build_matcher

import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    elif hasattr(criterion, 'loss_hoi_labels'):
        metric_logger.add_meter('hoi_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    preds = []
    gts = []
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device) #torch.Size([4, 3, 480, 854])
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]
        outputs = model(samples)
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        elif hasattr(criterion, 'loss_hoi_labels'):
            metric_logger.update(hoi_class_error=loss_dict_reduced['hoi_class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

## new
@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, criterion,
                 subject_category_id, device, args, detect, recognition, mAPi, detect_i):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    total = 0
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
      
        outputs= model(samples, is_training=False)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)
        
        format = "list"
        ## test
        # labels,predicts,len_gt = formatLabelstest_nogt(targets)
        labels, predicts, len_gt, count = formatLabels(targets, results)
        labels_i, predicts_i = formatLabels_ori(targets, results, len_gt)   
        total += count
        
        ivt_gt, pred_rec_hoi = recLabels(labels,predicts,len_gt)
        pred_rec_hoi = [result['hoi_scores'][0] for result in results]
        pred_rec_hoi = [[float(element) for element in sublist] for sublist in pred_rec_hoi]
        pred_rec_hoi = torch.tensor(pred_rec_hoi)
        ivt_gt = torch.tensor(ivt_gt)
        # recognition use first query
        i_gt = recLabels_i(labels, len_gt)
        pred_rec_i = [result['sub_scores'][0] for idx, result in enumerate(results.copy())]
        pred_rec_i = [[float(element) for element in sublist] for sublist in pred_rec_i]
        pred_rec_i = torch.tensor(pred_rec_i)
        pred_rec_i = pred_rec_i[:, :6]
        i_gt = torch.tensor(i_gt)
        # recognition 
        detect_i.update_frame(labels_i, predicts_i, format=format) #4，400
        detect.update_frame(labels, predicts, format=format) #4，400
        recognition.update(ivt_gt, pred_rec_hoi)
        mAPi.update(i_gt, pred_rec_i)
        
        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))
    recognition.video_end() 
    detect.video_end()
    detect_i.video_end()
    mAPi.video_end()
    results_rec_i = mAPi.compute_video_AP()
    results_rec_ivt = recognition.compute_video_AP('ivt')
    results_ivt = detect.compute_video_AP("ivt")
    results_i = detect.compute_video_AP("i")
    results_ivt1 = detect_i.compute_video_AP("ivt")
    results_i1 = detect_i.compute_video_AP("i")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    return results_ivt, results_i, results_rec_ivt, results_rec_i, results_ivt1, results_i1

def recLabels_i(labels,len_gt):
    gt = []
    new_length = 0
    for index in range(len(len_gt)):
        new_gt = [0] * 6
        old_length = new_length
        new_length = old_length + len_gt[index]
        for id in range(new_length-old_length):
            gt_id = int(labels[old_length + id][1])
            new_gt[gt_id] = 1
        gt.append(new_gt)
        
    return gt

def recLabels(labels,predicts,len_gt):
    gt = []
    pred = []
    new_length = 0
    for index in range(len(len_gt)):
        new_gt = [0] * 100
        new_pred = [0] * 100
        old_length = new_length
        new_length = old_length + len_gt[index]
        for id in range(new_length-old_length):
            gt_id = int(labels[old_length + id][0])
            new_gt[gt_id] = 1
            pred_id = int(predicts[old_length + id][0])
            new_pred[pred_id] = 1
        gt.append(new_gt)
        pred.append(new_pred)
        
    return gt, pred
 
def formatLabelstest_nogt(targets):
    
    labels = []
    predicts = []
    len_gt = []
    for target in targets:
        len_gt.append(len(target['hois']))
        hois = target['hois']
        gt_labels = target['labels']
        for index in range(len(hois)):
            label = []
            sub_index = hois[index][0]
            obj_index = hois[index][1]
            verb_id = int(hois[index][2])
            sub_id = int(gt_labels[sub_index])
            obj_id = int(gt_labels[obj_index])  
            triplet_id = search_id(sub_id, verb_id, obj_id)
            
            tool_id = sub_id
            tool_probs = float(1)
            label.append(triplet_id)
            label.append(tool_id)
            label.append(tool_probs)
            
            img_h, img_w = target['orig_size'].unbind(0)
            scale_fct = [img_w, img_h, img_w, img_h]
            box = target['boxes'][index].float() #xyxy
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            box = torch.stack([x.float() / y.float() for x, y in zip(box, scale_fct)])
            label.extend(box)
            label = [float(elem.item()) if isinstance(elem, torch.Tensor) else elem for elem in label]
            
            labels.append(label)
            
    for target in targets:
        pred_boxes = target['pred_sub_bbox']
        pred_ids = target['pred_sub_id']
        for index in range(len(pred_boxes)):
            predict = []
            sub_id = int(pred_ids[index])
            triplet_id = 0
            tool_id = sub_id
            tool_probs = float(1)
            predict.append(triplet_id)
            predict.append(tool_id)
            predict.append(tool_probs)
            
            img_h, img_w = target['orig_size'].unbind(0)
            scale_fct = [img_w, img_h, img_w, img_h] 
            
            box = pred_boxes[index].float().clone()
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            box = torch.stack([x.float() / y.float() for x, y in zip(box, scale_fct)])
                
            predict.extend(box)
            predict = [float(elem.item()) if isinstance(elem, torch.Tensor) else elem for elem in predict]
            predicts.append(predict)
    
    return labels,predicts,len_gt   
    
def formatLabelstest(targets):
    
    labels = []
    predicts = []
    len_gt = []
    for target in targets:
        len_gt.append(len(target['hois']))
        hois = target['hois']
        gt_labels = target['labels']
        for index in range(len(hois)):
            label = []
            sub_index = hois[index][0]
            obj_index = hois[index][1]
            verb_id = int(hois[index][2])
            sub_id = int(gt_labels[sub_index])
            obj_id = int(gt_labels[obj_index])  
            triplet_id = search_id(sub_id, verb_id, obj_id)
            
            tool_id = sub_id
            tool_probs = float(1)
            label.append(triplet_id)
            label.append(tool_id)
            label.append(tool_probs)
            
            img_h, img_w = target['orig_size'].unbind(0)
            scale_fct = [img_w, img_h, img_w, img_h]
            box = target['boxes'][index].float() #xyxy
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            box = torch.stack([x.float() / y.float() for x, y in zip(box, scale_fct)])
            label.extend(box)
            label = [float(elem.item()) if isinstance(elem, torch.Tensor) else elem for elem in label]
            
            labels.append(label)
            
    for target in targets:
        hois = target['hois']
        gt_labels = target['labels']
        for index in range(len(hois)):
            predict = []
            sub_index = hois[index][0]
            obj_index = hois[index][1]
            verb_id = int(hois[index][2])
            sub_id = int(gt_labels[sub_index])
            obj_id = int(gt_labels[obj_index])  
            triplet_id = search_id(sub_id, verb_id, obj_id)
            
            tool_id = sub_id
            tool_probs = float(1)
            predict.append(triplet_id)
            predict.append(tool_id)
            predict.append(tool_probs)
            
            img_h, img_w = target['orig_size'].unbind(0)
            scale_fct = [img_w, img_h, img_w, img_h] 
            ids = target['pred_sub_id']
            if target['pred_sub_bbox'] != [] and index <= len(target['pred_sub_bbox'])-1:
                boxes = target['pred_sub_bbox'].float().clone()
                for k, id in enumerate(ids):
                    if id == sub_id:
                        ids[k] = 100
                        box = boxes[k].float()
                        box[2] = box[2] - box[0]
                        box[3] = box[3] - box[1]
                        box = torch.stack([x.float() / y.float() for x, y in zip(box, scale_fct)])
                        break
            else:
                box = torch.zeros((4,))
                
            predict.extend(box)
            predict = [float(elem.item()) if isinstance(elem, torch.Tensor) else elem for elem in predict]
            predicts.append(predict)
    
    return labels,predicts,len_gt

def formatLabels_i(targets, results):
    
    labels = []
    predicts = []
    len_gt = []
    for target in targets:
        len_gt.append(len(target['hois']))
        hois = target['hois']
        gt_labels = target['labels']
        for index in range(len(hois)):
            label = []
            sub_index = hois[index][0]
            obj_index = hois[index][1]
            verb_id = int(hois[index][2])
            sub_id = int(gt_labels[sub_index])
            obj_id = int(gt_labels[obj_index])  
            triplet_id = search_id(sub_id, verb_id, obj_id)
            
            tool_id = sub_id
            tool_probs = float(1)
            label.append(triplet_id)
            label.append(tool_id)
            label.append(tool_probs)
            
            img_h, img_w = target['orig_size'].unbind(0)
            scale_fct = [img_w, img_h, img_w, img_h]
            box = target['boxes'][index].float() #xyxy
            box = torch.stack([x.float() / y.float() for x, y in zip(box, scale_fct)])
            label.extend(box)
            label = [float(elem.item()) if isinstance(elem, torch.Tensor) else elem for elem in label]
            
            labels.append(label)
            
    for index, result in enumerate(results):    
        max_subs, max_sub_col = torch.max(result['sub_scores'][:,:6], dim=1) 
        max_subs = torch.stack([max_subs, max_sub_col], dim=1) #torch.Size([100,2])
        
        max_scores, indices = torch.topk(max_subs[:, 0], len_gt[index], largest=True)
        max_subs = max_subs[indices]
        max_subs = [list(max_sub) for max_sub in max_subs]
        
        boxes = targets[index]['pred_sub_bbox']
        pred_ids = targets[index]['pred_sub_id']
        
        for hoi_id in range(len(max_subs)):
            sub_id = int(max_subs[hoi_id][1])
            flag = False
            for pred_id in range(len(pred_ids)):
                if pred_ids[pred_id] == sub_id:
                   max_subs[hoi_id].append(boxes[pred_id])
                   flag = True
                   break
            if flag == False:
                zero_box = torch.zeros((4,))
                max_subs[hoi_id].append(zero_box)
        
        for max_sub in max_subs:
            max_sub.append(0)
            max_sub[0] = 1
            predict = [convert_tensor_to_scalar(item) for item in max_sub]
            predict = flatten_list(predict)
            predict = predict[:2] + predict[6:7] + predict[2:6]  
            predicts.append(predict)
    
    return labels,predicts

##new how to handle when have two triplet or more ##
def formatLabels(targets, results):
    count = 0
    labels = []
    predicts = []
    len_gt = []
    for target in targets:
        len_gt.append(len(target['hois']))
        hois = target['hois']
        gt_labels = target['labels']
        for index in range(len(hois)):
            label = []
            sub_index = hois[index][0]
            obj_index = hois[index][1]
            verb_id = int(hois[index][2])
            sub_id = int(gt_labels[sub_index])
            obj_id = int(gt_labels[obj_index])  
            triplet_id = search_id(sub_id, verb_id, obj_id)
            
            tool_id = sub_id
            tool_probs = float(1)
            label.append(triplet_id)
            label.append(tool_id)
            label.append(tool_probs)
            
            img_h, img_w = target['orig_size'].unbind(0)
            scale_fct = [img_w, img_h, img_w, img_h]
            box = target['boxes'][index].float() #xyxy
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            box = torch.stack([x.float() / y.float() for x, y in zip(box, scale_fct)])
            label.extend(box)
            label = [float(elem.item()) if isinstance(elem, torch.Tensor) else elem for elem in label]
            
            labels.append(label)
        
    hico_triplet_labels = list(hico_text_label.keys())
    hoi_sub_list = []
    max_hois = 20
    for hoi_pair in hico_triplet_labels:
        hoi_sub_list.append(hoi_pair[0])
    for index, img_preds in enumerate(results):        
        img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()} 
        
        ##change sub_scores^2 to sub_scores
        sub_scores = img_preds['sub_scores'] *  img_preds['sub_scores'] # [100, 7]
        hoi_scores = img_preds['hoi_scores'] + 2 * sub_scores[:, hoi_sub_list] 

        hoi_scores = hoi_scores.ravel() # fallaten [10000]
        topk_hoi_scores = top_k(list(hoi_scores), max_hois) #list [20]
        topk_indexes = np.array([np.where(hoi_scores == score)[0][0] for score in topk_hoi_scores]) #[20]

        len_hoi_scores = img_preds['hoi_scores'].shape[1]
        quotients = topk_indexes // len_hoi_scores
        remainders = topk_indexes % len_hoi_scores
        top_hois = np.column_stack((quotients, remainders, topk_hoi_scores)) #row, column, score
        ## trick
        ## filter same query
        row_hois = []
        hoi_ids = []
        for top_hoi in top_hois:
            hoi_id = top_hoi[0]
            if hoi_id not in hoi_ids:
                hoi_ids.append(hoi_id)
                row_hois.append(top_hoi)   
        final_hois = row_hois    
        ## add subject id     
        for id in range(len(final_hois)):
            final_hois[id] = final_hois[id].tolist()
            sub_id = search_subid(final_hois[id][1])
            final_hois[id].append(sub_id)
        ## get same length hoi
        while len(final_hois) < len_gt[index]:
            add = final_hois[0].copy()
            final_hois.append(add)
        if len(final_hois) >= len_gt[index]:
            final_hois = final_hois[:len_gt[index]]      
        ## get scaled bbox and pred_id
        boxes = targets[index]['pred_sub_bbox']
        img_h, img_w = targets[index]['orig_size'].unbind(0)
        scale_fct = [img_w, img_h, img_w, img_h]
        for box_id in range(len(boxes)):
            boxes[box_id][2] = boxes[box_id][2] - boxes[box_id][0]
            boxes[box_id][3] = boxes[box_id][3] - boxes[box_id][1]
            boxes[box_id] = torch.stack([x.float() / y.float() for x, y in zip(boxes[box_id], scale_fct)])
        pred_ids = targets[index]['pred_sub_id']
        ## add scaled box
        for hoi_id in range(len(final_hois)):
            sub_id = final_hois[hoi_id][3]
            flag = False
            for pred_id in range(len(pred_ids)):
                if pred_ids[pred_id] == sub_id:
                   count += 1
                   final_hois[hoi_id].append(boxes[pred_id])
                   flag = True
                   break
            if flag == False:
                zero_box = torch.zeros((4,))
                final_hois[hoi_id].append(zero_box)
                      
        for final_hoi in final_hois:
            predict = [elem.tolist() if isinstance(elem, torch.Tensor) else elem for elem in final_hoi]
            predict = flatten_list(predict)
            predict = predict[ 1:]
            predict = predict[:1] + predict[2:3] + predict[1:2] + predict[3:] 
            predicts.append(predict)
    return labels,predicts,len_gt,count

def formatLabels_ori(targets, results, len_gt):
    labels = []
    predicts = []
    for target in targets:
        hois = target['hois']
        gt_labels = target['labels']
        for index in range(len(hois)):
            label = []
            sub_index = hois[index][0]
            obj_index = hois[index][1]
            verb_id = int(hois[index][2])
            sub_id = int(gt_labels[sub_index])
            obj_id = int(gt_labels[obj_index])  
            triplet_id = search_id(sub_id, verb_id, obj_id)
            
            tool_id = sub_id
            tool_probs = float(1)
            label.append(triplet_id)
            label.append(tool_id)
            label.append(tool_probs)
            
            img_h, img_w = target['orig_size'].unbind(0)
            scale_fct = [img_w, img_h, img_w, img_h]
            box = target['boxes'][index].float() #xyxy
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            box = torch.stack([x.float() / y.float() for x, y in zip(box, scale_fct)])
            label.extend(box)
            label = [float(elem.item()) if isinstance(elem, torch.Tensor) else elem for elem in label]
            
            labels.append(label)
        
    hico_triplet_labels = list(hico_text_label.keys())
    hoi_sub_list = []
    max_hois = 20
    for hoi_pair in hico_triplet_labels:
        hoi_sub_list.append(hoi_pair[0])
    for index, img_preds in enumerate(results):
        sub_boxes = img_preds['sub_boxes'] #torch.Size([100, 4]) #cxcywh
        sub_boxes = box_cxcywh_to_xywh(sub_boxes)
        
        img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()} 
        
        sub_scores = img_preds['sub_scores'] * img_preds['sub_scores'] # [100, 7]
        hoi_scores = img_preds['hoi_scores'] + 2 * sub_scores[:, hoi_sub_list] 

        hoi_scores = hoi_scores.ravel() # fallaten [10000]
        topk_hoi_scores = top_k(list(hoi_scores), max_hois) #list [20]
        topk_indexes = np.array([np.where(hoi_scores == score)[0][0] for score in topk_hoi_scores]) #[20]

        len_hoi_scores = img_preds['hoi_scores'].shape[1]
        quotients = topk_indexes // len_hoi_scores
        remainders = topk_indexes % len_hoi_scores
        top_hois = np.column_stack((quotients, remainders, topk_hoi_scores)) #row, column, score
        
        final_hois = []
        hoi_ids = []
        for top_hoi in top_hois:
            hoi_id = top_hoi[0]
            if hoi_id not in hoi_ids:
                hoi_ids.append(hoi_id)
                final_hois.append(top_hoi)   
        ## add subject id     
        for id in range(len(final_hois)):
            final_hois[id] = final_hois[id].tolist()
            sub_id = search_subid(final_hois[id][1])
            final_hois[id].append(sub_id)
            sub_box = sub_boxes[int(final_hois[id][0])]
            final_hois[id].append(sub_box)
        ## get same length hoi
        while len(final_hois) < len_gt[index]:
            add = final_hois[0].copy()
            final_hois.append(add)
        if len(final_hois) >= len_gt[index]:
            final_hois = final_hois[:len_gt[index]] 
        
        for final_hoi in final_hois:
            predict = [elem.tolist() if isinstance(elem, torch.Tensor) else elem for elem in final_hoi]
            predict = flatten_list(predict)
            predict = predict[1:]
            predict = predict[:1] + predict[2:3] + predict[1:2] + predict[3:] 
            predicts.append(predict)

    return labels, predicts

        

def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def search_subid(hoi_id):
    combinations = hico_text_label
    triplet = list(combinations.keys())[int(hoi_id)]
    sub_id = triplet[0]
    
    return sub_id

        
def search_id(sub_id, verb_id, obj_id):
    given_combination = (sub_id, verb_id, obj_id)
    combinations = hico_text_label

    index = 1
    for combination in combinations:
        if combination == given_combination:
            return index - 1
        index += 1
    
def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), w, h]
    return torch.stack(b, dim=-1)

def nms(boxes, scores, threshold=0.5):
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlaps = w * h

        # IoU
        ious = overlaps / (areas[i] + areas[order[1:]] - overlaps)

        # IOU<threshold , remove
        idx = np.where(ious <= threshold)[0]
        order = order[idx + 1]
        
    selected_boxes = boxes[keep]
    selected_scores = scores[keep]

    return selected_boxes, selected_scores

def convert_tensor_to_scalar(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.numel() == 1:
            return tensor.item()
        else:
            return tensor.tolist()
    else:
        return tensor
    