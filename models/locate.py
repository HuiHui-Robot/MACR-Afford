import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dino import vision_transformer as vits
from models.dino.utils import load_pretrained_weights
from models.model_util import *
from fast_pytorch_kmeans import KMeans
from utils.attention import *
from typing import Any, Union, List
from pkg_resources import packaging
from simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
import json
from collections import OrderedDict

_tokenizer = _Tokenizer()


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


def llm_file():
    llm_path = "./CLIP/AGD20K_Affordance_object_sentence.json"
    with open(llm_path, "r") as file:
        GPT_prompt_dict = json.load(file)
    return GPT_prompt_dict


class PromptLearner(nn.Module):
    def __init__(self, classnames, ln_final, token_embedding, divide):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = ln_final.weight.dtype
        ctx_dim = ln_final.weight.shape[0]
        
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        if divide == "Seen":
            Affordance_object = llm_file()
            prompts = [prompt_prefix + " " + Affordance_object[name] for name in classnames]
        else:
            Affordance_object = llm_file()
            prompts = [prompt_prefix + " " + Affordance_object[name] for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = 'end'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
    torch.IntTensor, torch.LongTensor]:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x_1 = x + self.positional_embedding[:, None, :].to(x.dtype)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, att = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )
        return x[0], x[1:], att[:, 1:, 1:]


class Net(nn.Module):

    def __init__(self, args, embed_dim: int, context_length: int, vocab_size: int,
                 transformer_width: int, transformer_heads: int,
                 transformer_layers: int, num_classes=36, pretrained=True, n=3, D=512, divide="Seen"):
        super(Net, self).__init__()

        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.pretrained = pretrained
        self.n = n
        self.D = D
        self.context_length = context_length

        self.classnames = ['beat', "boxing", "brush_with", "carry", "catch",
                           "cut", "cut_with", "drag", 'drink_with', "eat",
                           "hit", "hold", "jump", "kick", "lie_on", "lift",
                           "look_out", "open", "pack", "peel", "pick_up",
                           "pour", "push", "ride", "sip", "sit_on", "stick",
                           "stir", "swing", "take_photo", "talk_on", "text_on",
                           "throw", "type_on", "wash", "write"]

        self.vit_feat_dim = 384
        self.cluster_num = 3
        self.stride = 16
        self.patch = 16
        self.divide = divide

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.attnpool = AttentionPool2d(14, self.vit_feat_dim, 64, embed_dim)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.prompt_learner = PromptLearner(self.classnames, self.ln_final, self.token_embedding, self.divide)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.aff_classes = num_classes
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.aff_cam_thd = 0.6
        self.part_iou_thd = 0.6
        self.cel_margin = 0.5

        self.vit_model = vits.__dict__['vit_small'](patch_size=self.patch, num_classes=0)
        load_pretrained_weights(self.vit_model, '', None, 'vit_small', self.patch)

        self.fc0 = nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(self.vit_feat_dim)

        self.linear = nn.Conv2d(self.vit_feat_dim, self.aff_classes, kernel_size=1, stride=1, padding=0)

        self.aff_proj3 = Mlp(in_features=int(self.vit_feat_dim * 3), hidden_features=int(self.vit_feat_dim * 3),
                             out_features=self.vit_feat_dim,
                             act_layer=nn.GELU, drop=0.)
        self.aff_proj = Mlp(in_features=self.vit_feat_dim, hidden_features=int(self.vit_feat_dim * 4),
                            act_layer=nn.GELU, drop=0.)

        self.aff_ego_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_exo_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )

        self.aff_fc = nn.Conv2d(self.vit_feat_dim, self.aff_classes, 1)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def encode_text(self, per, text):
        x = per + self.token_embedding(text.cuda()).float()

        x = x + self.positional_embedding.float()
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).float()

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def CPCL(self, out3, out4, out5):
        out5a = F.relu(self.bn0(self.fc0(out5.mean(dim=(2, 3), keepdim=True))), inplace=True)
        out4a = F.relu(self.bn0(self.fc0(out4.mean(dim=(2, 3), keepdim=True))), inplace=True)
        out3a = F.relu(self.bn0(self.fc0(out3.mean(dim=(2, 3), keepdim=True))), inplace=True)

        vector = out5a * out4a * out3a

        out5b = torch.sigmoid(self.fc0(vector)) * out5
        out4b = torch.sigmoid(self.fc0(vector)) * out4
        out3b = torch.sigmoid(self.fc0(vector)) * out3

        p1 = F.relu(self.linear(out5b), inplace=True)
        p2 = F.relu(self.linear(out4b), inplace=True)
        p3 = F.relu(self.linear(out3b), inplace=True)

        outb = out5b * out4b * out3b

        drop5 = attention_drop(p1, out5)
        drop4 = attention_drop(p2, out4)
        drop3 = attention_drop(p3, out3)

        out5ap = F.relu(self.bn0(self.fc0(drop5.mean(dim=(2, 3), keepdim=True))), inplace=True)
        out4ap = F.relu(self.bn0(self.fc0(drop4.mean(dim=(2, 3), keepdim=True))), inplace=True)
        out3ap = F.relu(self.bn0(self.fc0(drop3.mean(dim=(2, 3), keepdim=True))), inplace=True)

        vectorp = out5ap * out4ap * out3ap

        out5p = torch.sigmoid(self.fc0(vectorp)) * drop5
        out4p = torch.sigmoid(self.fc0(vectorp)) * drop4
        out3p = torch.sigmoid(self.fc0(vectorp)) * drop3

        outp = out5p * out4p * out3p

        pred2 = F.dropout(outb, p=0.5)
        pred2 = self.aff_fc(pred2)

        pred4 = F.dropout(outp, p=0.5)
        pred4 = self.aff_fc(pred4)

        pred = torch.maximum(pred2, pred4)

        return pred


    def forward(self, exo, ego, aff_label, epoch):
        num_exo = exo.shape[1]
        exo = exo.flatten(0, 1)

        with torch.no_grad():
            _, exo_key, exo_attn = self.vit_model.get_last_three_keys(exo)
            exo_desc_temp = []
            for key in exo_key:
                desc = key.permute(0, 2, 3, 1).flatten(-2, -1).detach()
                exo_desc_temp.append(desc)

        exo_proj = []
        for desc in exo_desc_temp:
            exo_proj_temp = desc[:, 1:] + self.aff_proj(desc[:, 1:])
            exo_proj_temp = self._reshape_transform(exo_proj_temp, self.patch, self.stride)
            exo_proj.append(exo_proj_temp)

        with torch.no_grad():
            _, ego_key, ego_attn = self.vit_model.get_all_key(ego)
            _, exo_key, exo_attn = self.vit_model.get_all_key(exo)

            ego_desc = ego_key[len(ego_key) - 3].permute(0, 2, 3, 1).flatten(-2, -1).detach()
            exo_desc = exo_key[len(ego_key) - 3].permute(0, 2, 3, 1).flatten(-2, -1).detach()

            for i in range(len(ego_key) - 2, len(ego_key)):
                ego_desc = torch.cat((ego_desc, ego_key[i].permute(0, 2, 3, 1).flatten(-2, -1).detach()), dim=2)
                exo_desc = torch.cat((exo_desc, exo_key[i].permute(0, 2, 3, 1).flatten(-2, -1).detach()), dim=2)

        ego_proj = self.aff_proj3(ego_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        exo_desc = self._reshape_transform(exo_desc[:, 1:, :], self.patch, self.stride)

        aff_cam = self.CPCL(exo_proj[0], exo_proj[1], exo_proj[2])

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.encode_text(prompts, tokenized_prompts)

        e_b, e_c, e_h, e_w = ego_proj.shape
        pre_ego = ego_proj
        image_features, ego_proj, mu_att = self.attnpool(ego_proj)

        image_features = F.normalize(image_features, dim=1, p=2)
        text_features = F.normalize(text_features, dim=1, p=2)

        logit_scale = self.logit_scale.exp()

        self.logits_per_image = logit_scale * image_features @ text_features.t()
        self.logits_per_text = self.logits_per_image.t()
        loss_func = nn.CrossEntropyLoss().cuda()
        loss_i = loss_func(self.logits_per_image, aff_label)

        text_f = torch.ones((e_b, 1024)).cuda()
        for i in range(e_b):
            text_f[i] = text_features[aff_label[i]]
        att_egoproj = F.normalize(ego_proj, dim=1, p=2)
        attego = logit_scale * att_egoproj.permute(1, 0, 2) @ text_f.unsqueeze(2)
        attego = torch.sigmoid(F.normalize(attego, dim=1, p=2)).permute(1, 0, 2).repeat(1, 1, e_c)
        ego_proj = attego.permute(1, 2, 0).view(e_b, e_c, e_h, e_w) * pre_ego + pre_ego

        b, c, h, w = ego_desc.shape
        ego_cls_attn = ego_attn[:, :, 0, 1:].reshape(b, 6, h, w)
        ego_cls_attn = (ego_cls_attn > ego_cls_attn.flatten(-2, -1).mean(-1, keepdim=True).unsqueeze(-1)).float()
        head_idxs = [0, 1, 3]
        ego_sam = ego_cls_attn[:, head_idxs].mean(1)
        ego_sam = normalize_minmax(ego_sam)
        ego_sam_flat = ego_sam.flatten(-2, -1)

        aff_logits = self.gap(aff_cam).reshape(b, num_exo, self.aff_classes)

        aff_cam_re = aff_cam.reshape(b, num_exo, self.aff_classes, h, w)

        gt_aff_cam = torch.zeros(b, num_exo, h, w).cuda()
        for b_ in range(b):
            gt_aff_cam[b_, :] = aff_cam_re[b_, :, aff_label[b_]]

        ego_desc_flat = ego_desc.flatten(-2, -1)
        exo_desc_re_flat = exo_desc.reshape(b, num_exo, c, h, w).flatten(-2, -1)
        sim_maps = torch.zeros(b, self.cluster_num, h * w).cuda()
        exo_sim_maps = torch.zeros(b, num_exo, self.cluster_num, h * w).cuda()
        part_score = torch.zeros(b, self.cluster_num).cuda()
        part_proto = torch.zeros(b, c).cuda()
        for b_ in range(b):
            exo_aff_desc = []
            for n in range(num_exo):
                tmp_cam = gt_aff_cam[b_, n].reshape(-1)
                tmp_max, tmp_min = tmp_cam.max(), tmp_cam.min()
                tmp_cam = (tmp_cam - tmp_min) / (tmp_max - tmp_min + 1e-10)
                tmp_desc = exo_desc_re_flat[b_, n]
                tmp_top_desc = tmp_desc[:, torch.where(tmp_cam > self.aff_cam_thd)[0]].T
                exo_aff_desc.append(tmp_top_desc)
            exo_aff_desc = torch.cat(exo_aff_desc, dim=0)

            if exo_aff_desc.shape[0] < self.cluster_num:
                continue

            kmeans = KMeans(n_clusters=self.cluster_num, mode='euclidean', max_iter=300)
            kmeans.fit_predict(exo_aff_desc.contiguous())

            clu_cens = F.normalize(kmeans.centroids, dim=1)

            for n_ in range(num_exo):
                exo_sim_maps[b_, n_] = torch.mm(clu_cens, F.normalize(exo_desc_re_flat[b_, n_], dim=0))

            sim_map = torch.mm(clu_cens, F.normalize(ego_desc_flat[b_], dim=0))
            tmp_sim_max, tmp_sim_min = torch.max(sim_map, dim=-1, keepdim=True)[0], torch.min(sim_map, dim=-1, keepdim=True)[0]

            sim_map_norm = (sim_map - tmp_sim_min) / (tmp_sim_max - tmp_sim_min + 1e-12)

            sim_map_hard = (sim_map_norm > torch.mean(sim_map_norm, 1, keepdim=True)).float()

            sam_hard = (ego_sam_flat > torch.mean(ego_sam_flat, 1, keepdim=True)).float()

            inter = (sim_map_hard * sam_hard[b_]).sum(1)
            union = sim_map_hard.sum(1) + sam_hard[b_].sum() - inter

            p_score = (inter / sim_map_hard.sum(1) + sam_hard[b_].sum() / union) / 2

            sim_maps[b_] = sim_map
            part_score[b_] = p_score

            if p_score.max() < self.part_iou_thd:
                continue
            part_proto[b_] = clu_cens[torch.argmax(p_score)]

        sim_maps = sim_maps.reshape(b, self.cluster_num, h, w)
        exo_sim_maps = exo_sim_maps.reshape(b, num_exo, self.cluster_num, h, w)

        ego_proj = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj)
        aff_logits_ego = self.gap(ego_pred).view(b, self.aff_classes)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        loss_con = torch.zeros(1).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]
            loss_con += concentration_loss(ego_pred[b_])

        gt_ego_cam = normalize_minmax(gt_ego_cam)

        loss_con /= b

        loss_proto = torch.zeros(1).cuda()
        valid_batch = 0
        if epoch[0] > epoch[1]:
            for b_ in range(b):
                if not part_proto[b_].equal(torch.zeros(c).cuda()):
                    mask = gt_ego_cam[b_]
                    tmp_feat = ego_desc[b_] * mask
                    embedding = tmp_feat.reshape(tmp_feat.shape[0], -1).sum(1) / mask.sum()
                    loss_proto += torch.max(
                        1 - F.cosine_similarity(embedding, part_proto[b_], dim=0) - self.cel_margin,
                        torch.zeros(1).cuda())
                    valid_batch += 1
            loss_proto = loss_proto / (valid_batch + 1e-15)

        masks = {'exo_aff': gt_aff_cam, 'ego_sam': ego_sam,
                'pred': (sim_maps, exo_sim_maps, part_score, gt_ego_cam)}
        logits = {'aff': aff_logits, 'aff_ego': aff_logits_ego}

        return masks, logits, loss_proto, loss_con, loss_i


    @torch.no_grad()
    def test_forward(self, ego, aff_label):
        _, ego_key, ego_attn = self.vit_model.get_all_key(ego)
        ego_desc = ego_key[len(ego_key) - 3].permute(0, 2, 3, 1).flatten(-2, -1).detach()
        for i in range(len(ego_key) - 2, len(ego_key)):
            ego_desc = torch.cat((ego_desc, ego_key[i].permute(0, 2, 3, 1).flatten(-2, -1).detach()), dim=2)

        ego_proj = self.aff_proj3(ego_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.encode_text(prompts, tokenized_prompts)

        e_b, e_c, e_h, e_w = ego_proj.shape
        pre_ego = ego_proj
        image_features, ego_proj, mu_att = self.attnpool(ego_proj)

        text_features = F.normalize(text_features, dim=1, p=2)

        logit_scale = self.logit_scale.exp()

        text_f = torch.ones((e_b, 1024)).cuda()
        for i in range(e_b):
            text_f[i] = text_features[aff_label[i]]
        att_egoproj = F.normalize(ego_proj, dim=1, p=2)
        attego = logit_scale * att_egoproj.permute(1, 0, 2) @ text_f.unsqueeze(2)
        attego = torch.sigmoid(F.normalize(attego, dim=1, p=2)).permute(1, 0, 2).repeat(1, 1, e_c)
        ego_proj = attego.permute(1, 2, 0).view(e_b, e_c, e_h, e_w) * pre_ego + pre_ego

        b, c, h, w = ego_desc.shape
        ego_proj = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]

        return gt_ego_cam


    def _reshape_transform(self, tensor, patch_size, stride):
        height = (224 - patch_size) // stride + 1
        width = (224 - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.transpose(2, 3).transpose(1, 2).contiguous()
        return result

def MODEL(args, num_classes=36,
          pretrained=True, n=3, D=512, divide="Seen"):
    dict = 'RN50.pt'  
    state_dict = torch.jit.load(dict)
    state_dict = state_dict.state_dict()
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = Net(args, embed_dim=embed_dim, context_length=context_length, vocab_size=vocab_size,
                transformer_width=transformer_width,
                transformer_heads=transformer_heads, transformer_layers=transformer_layers, num_classes=num_classes,
                pretrained=pretrained, n=n, D=D, divide=divide)

    model_dict = model.state_dict()
    par = []
    pretrained_dict = {}
    for para in model.named_parameters():
        k = para[0]
        if k in state_dict:
            par.append(para[0])
    for k, v in state_dict.items():
        if k in model_dict:
            pretrained_dict[k] = v

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model, par