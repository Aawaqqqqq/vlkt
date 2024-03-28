import torch
from torch import nn
import torch.nn.functional as F
import pprint

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
import clip
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label

from .backbone import build_backbone
from .matcher import build_matcher
from .gen import build_gen
# from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

# model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
# tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


class GEN_VLKT(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        super().__init__()

        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.total_input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        ##new
        self.sub_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                
        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label
            unseen_index = None

        # torch.Size([100, 512]) #torch.Size([22, 512]) #torch.Size([768, 512]) ## #torch.Size([22, 77]) #torch.Size([100, 512])
        self.clip_label, obj_clip_label, v_linear_proj_weight, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_CLIP(hoi_text_label, obj_text_label, unseen_index)
        ##new 
        sub_clip_label = torch.cat((obj_clip_label[0:6, :], torch.unsqueeze(obj_clip_label[22, :], dim=0)), dim=0)
        obj_clip_label = obj_clip_label[6:22, :]
        num_obj_classes = 16
        num_sub_classes = 7

        self.hoi_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, args.clip_embed_dim),
            nn.LayerNorm(args.clip_embed_dim),
        )

        if args.with_clip_label:
            self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text)) #Linear(in_features=512, out_features=100, bias=True)
            self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True) #torch.Size([100, 512])
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
                self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600)
                self.eval_visual_projection.weight.data = self.clip_label / self.clip_label.norm(dim=-1, keepdim=True)
        else:
            self.hoi_class_embedding = nn.Linear(args.clip_embed_dim, len(hoi_text))

        if args.with_obj_clip_label:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )
            self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes)
            self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
            
            self.sub_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )
            self.sub_visual_projection = nn.Linear(args.clip_embed_dim, num_sub_classes)
            self.sub_visual_projection.weight.data = sub_clip_label / sub_clip_label.norm(dim=-1, keepdim=True)           
        else:
            self.new_obj_class_embed = nn.Linear(hidden_dim, num_obj_classes)
            self.new_sub_class_embed = nn.Linear(hidden_dim, num_sub_classes)

        self.hidden_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

    def init_classifier_with_CLIP(self, hoi_text_label, obj_text_label, unseen_index):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_inputs = torch.cat([clip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])
        if self.args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
        else:
            hoi_text_label_del = hoi_text_label.copy()
        text_inputs_del = torch.cat(
            [clip.tokenize(hoi_text_label[id]) for id in hoi_text_label_del.keys()])

        obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in obj_text_label])
        clip_model, preprocess = clip.load(self.args.clip_model, device=device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text_inputs.to(device))
            text_embedding_del = clip_model.encode_text(text_inputs_del.to(device))
            obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))
            v_linear_proj_weight = clip_model.visual.proj.detach()

        del clip_model

        return text_embedding.float(), obj_text_embedding.float(), v_linear_proj_weight.float(), \
               hoi_text_label_del, obj_text_inputs, text_embedding_del.float()
    
                                            
    def forward(self, samples: NestedTensor, is_training=True):
    ## def forward(self, samples: NestedTensor, is_training=True):
    # samples: torch.Size([4, 9, 747, 1332]) or torch.Size([4, 3, 3, 747, 1332])
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        b, t, c, h, w = samples.tensors.size()
        samples.tensors = samples.tensors.view(b*t, c, h, w)
        union_features, pos = self.backbone(samples) #torch.Size([12, 2048, 23, 41])  torch.Size([12, 256, 23, 41])  
        _, c, h, w = union_features[-1].tensors.size()
        features, mask = union_features[-1].decompose()
        
        features = features.view(b, t, c, h, w)
        src = features[:, 1:2, :, :, :].view(b, c, h, w) #torch.Size([4, 2048, 23, 41])
        assert mask is not None
        
        h_hs, o_hs, inter_hs = self.transformer(self.input_proj(src), mask,
                                                self.query_embed_h.weight,
                                                self.query_embed_o.weight,
                                                self.pos_guided_embedd.weight,
                                                pos[-1], self.clip_label,
                                                self.total_input_proj(features.permute(0, 2, 1, 3, 4)))[:3]

        # torch.Size([3, 4, 100, 256]) torch.Size([3, 4, 100, 256]) torch.Size([3, 4, 100, 256])
        outputs_sub_coord = self.hum_bbox_embed(h_hs).sigmoid() 
        outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()

        if self.args.with_obj_clip_label:
            obj_logit_scale = self.obj_logit_scale.exp() # tensor(14.2857, device='cuda:0', grad_fn=<ExpBackward0>)
            o_hs = self.obj_class_fc(o_hs) #torch.Size([3, 4, 100, 512])
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
            
            sub_logit_scale = self.sub_logit_scale.exp()
            h_hs = self.sub_class_fc(h_hs)
            h_hs = h_hs / h_hs.norm(dim=-1, keepdim=True)
            outputs_sub_class = sub_logit_scale * self.sub_visual_projection(h_hs)
        else:
            outputs_sub_class = self.new_sub_class_embed(h_hs)
            outputs_obj_class = self.new_obj_class_embed(o_hs)
        
        if self.args.with_clip_label:
            logit_scale = self.logit_scale.exp()
            inter_hs = self.hoi_class_fc(inter_hs)
            outputs_inter_hs = inter_hs.clone() #torch.Size([3, 4, 100, 512])
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' \
                    and (self.args.eval or not is_training):
                outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
            else:
                outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
        else:
            inter_hs = self.hoi_class_fc(inter_hs) #torch.Size([3, 4, 100, 512])
            outputs_inter_hs = inter_hs.clone()
            outputs_hoi_class = self.hoi_class_embedding(inter_hs) #torch.Size([3, 4, 100, 56])

        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_sub_logits': outputs_sub_class[-1],
               'pred_obj_logits': outputs_obj_class[-1], 'pred_sub_boxes': outputs_sub_coord[-1], 
               'pred_obj_boxes': outputs_obj_coord[-1]}
        
        if self.args.with_mimic:
            out['inter_memory'] = outputs_inter_hs[-1] #torch.Size([4, 100, 512])
        if self.aux_loss:
            if self.args.with_mimic:
                aux_mimic = outputs_inter_hs
            else:
                aux_mimic = None
                
            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_sub_class,
                                                            outputs_obj_class, outputs_sub_coord, 
                                                            outputs_obj_coord, aux_mimic)
        return out
        # return out

    @torch.jit.unused
    ##new
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_sub_class,outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord, outputs_inter_hs=None):

        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-self.dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_sub_logits': outputs_sub_class[-self.dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1]}
        if outputs_inter_hs is not None:
            aux_outputs['inter_memory'] = outputs_inter_hs[-self.dec_layers: -1]
        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # empty_obj_weight = torch.ones(self.num_obj_classes + 1)
        # empty_obj_weight[-1] = self.eos_coef
        # self.register_buffer('empty_obj_weight', empty_obj_weight)
        ##new
        empty_obj_weight = torch.ones(16)
        self.register_buffer('empty_obj_weight', empty_obj_weight)
        empty_sub_weight = torch.ones(7)
        empty_sub_weight[-1] = self.eos_coef
        self.register_buffer('empty_sub_weight', empty_sub_weight)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(args.clip_model, device=device)
        self.alpha = args.alpha

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits'] #torch.Size([4, 100, 15])

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)]) #torch.Size([4])
        target_classes_o = target_classes_o - 6 
        # target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        ##new
        target_classes = torch.full(src_logits.shape[:2], 15,
                                    dtype=torch.int64, device=src_logits.device)
        
        target_classes[idx] = target_classes_o  #torch.Size([4, 100])
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_obj_weight) #torch.Size([15])
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    ##new
    def loss_sub_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_sub_logits' in outputs
        sub_src_logits = outputs['pred_sub_logits'] #torch.Size([4, 100, 7])

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['sub_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(sub_src_logits.shape[:2], 6,
                                    dtype=torch.int64, device=sub_src_logits.device)
        target_classes[idx] = target_classes_o #torch.Size([4, 100])
        loss_sub_ce = F.cross_entropy(sub_src_logits.transpose(1, 2), target_classes, self.empty_sub_weight) #torch.Size([7])
        losses = {'loss_sub_ce': loss_sub_ce}
        if log:
            losses['sub_class_error'] = 100 - accuracy(sub_src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, topk=5):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits'] #torch.Size([4, 100, 56])

        idx = self._get_src_permutation_idx(indices) #(tensor([0, 1, 2, 3]), tensor([87, 16, 18, 68]))
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)]) #torch.Size([4, 56])
        target_classes = torch.zeros_like(src_logits) #torch.Size([4, 100, 56])
        target_classes[idx] = target_classes_o #torch.Size([4, 100, 56])
        
        src_logits = _sigmoid(src_logits) #torch.Size([4, 100, 56])
        loss_hoi_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha) #alpha = 0.5
        
        losses = {'loss_hoi_labels': loss_hoi_ce}

        _, pred = src_logits[idx].topk(topk, 1, True, True) #top5 possible indice
        acc = 0.0
        for tid, target in enumerate(target_classes_o):
            tgt_idx = torch.where(target == 1)[0]
            if len(tgt_idx) == 0:
                continue
            acc_pred = 0.0
            for tgt_rel in tgt_idx:
                acc_pred += (tgt_rel in pred[tid])
            acc += acc_pred / len(tgt_idx)
        rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        losses['hoi_class_error'] = torch.from_numpy(np.array(
            rel_labels_error)).to(src_logits.device).float()
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                       exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def mimic_loss(self, outputs, targets, indices, num_interactions):
        src_feats = outputs['inter_memory']
        src_feats = torch.mean(src_feats, dim=1)

        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
        loss_feat_mimic = F.l1_loss(src_feats, target_clip_feats)
        losses = {'loss_feat_mimic': loss_feat_mimic}
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                'obj_labels': self.loss_obj_labels,
                'sub_labels': self.loss_sub_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
                'feats_mimic': self.mimic_loss
            }
        else:
            loss_map = {
                'obj_labels': self.loss_obj_labels,
                'obj_cardinality': self.loss_obj_cardinality,
                'verb_labels': self.loss_verb_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']
        ##new 
        out_sub_logits = outputs['pred_sub_logits']
        ##

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]
        obj_labels = obj_labels + 6
        ##new
        sub_scores = out_sub_logits.sigmoid()
        sub_labels = F.softmax(out_sub_logits, -1)[..., :-1].max(-1)[1]
        ##
        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, ss, sl, os, ol, sb, ob, out_sb = hoi_scores[index], sub_scores[index], sub_labels[index], obj_scores[
                index],obj_labels[index], sub_boxes[index], obj_boxes[index], out_sub_boxes[index]
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu'), 'sub_boxes': out_sb.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update({'hoi_scores': hs.to('cpu'), 'sub_scores': ss.to('cpu'), 'obj_scores': os.to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})    
        

        return results


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    gen = build_gen(args)

    model = GEN_VLKT(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}

    weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    ##new 
    weight_dict['loss_sub_ce'] = args.sub_loss_coef 

    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.with_mimic:
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['hoi_labels', 'sub_labels', 'obj_labels', 'sub_obj_boxes']

    if args.with_mimic:
        losses.append('feats_mimic')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors