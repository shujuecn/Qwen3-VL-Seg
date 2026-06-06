import torch
import torch.nn as nn
import torch.nn.functional as F


class TongueMaskHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, mask_size=256):
        super().__init__()
        self.mask_size = mask_size
        self.proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, features, box_gate):
        features = self.proj(features)
        box_gate = F.interpolate(
            box_gate,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        logits = self.net(torch.cat([features * box_gate, box_gate], dim=1))
        return F.interpolate(
            logits,
            size=(self.mask_size, self.mask_size),
            mode="bilinear",
            align_corners=False,
        )


class Qwen3VLSegForConditionalGeneration(nn.Module):
    def __init__(self, base_model, seg_mask_size=256, seg_loss_weight=1.0):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.seg_mask_size = seg_mask_size
        self.seg_loss_weight = seg_loss_weight
        vision_config = base_model.config.vision_config
        in_dim = getattr(vision_config, "out_hidden_size", None) or vision_config.hidden_size
        self.mask_head = TongueMaskHead(in_dim=in_dim, mask_size=self.seg_mask_size)

        for param in self.base_model.parameters():
            param.requires_grad = False
        base_param = next(self.base_model.parameters())
        self.mask_head.to(device=base_param.device, dtype=base_param.dtype)

    def _image_features_to_tensor(self, image_features):
        if hasattr(image_features, "pooler_output"):
            return image_features.pooler_output
        if hasattr(image_features, "last_hidden_state"):
            return image_features.last_hidden_state
        if isinstance(image_features, (tuple, list)):
            return image_features[0]
        return image_features

    def _split_image_features(self, image_features, image_grid_thw):
        features = []
        offset = 0
        merge_size = getattr(self.base_model.config.vision_config, "spatial_merge_size", 2)
        for idx, grid in enumerate(image_grid_thw):
            t, h, w = [int(v) for v in grid.tolist()]
            h = h // merge_size
            w = w // merge_size
            length = t * h * w
            if isinstance(image_features, (tuple, list)):
                feat = image_features[idx]
            else:
                feat = image_features[offset : offset + length]
            if feat.shape[0] != length:
                raise ValueError(
                    f"image feature length mismatch: expected {length}, got {feat.shape[0]}"
                )
            if t != 1:
                feat = feat.view(t, h, w, -1).mean(dim=0)
            else:
                feat = feat.view(h, w, -1)
            features.append(feat.permute(2, 0, 1))
            offset += length
        return torch.stack(features)

    def _box_gate(self, boxes, size):
        bsz = boxes.shape[0]
        device = boxes.device
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, size, device=device),
            torch.linspace(0, 1, size, device=device),
            indexing="ij",
        )
        xx = xx[None].expand(bsz, -1, -1)
        yy = yy[None].expand(bsz, -1, -1)
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        gate = (
            (xx >= x1[:, None, None])
            & (xx <= x2[:, None, None])
            & (yy >= y1[:, None, None])
            & (yy <= y2[:, None, None])
        )
        return gate[:, None].to(dtype=torch.float32)

    def _dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        return (1 - (2 * intersection + 1.0) / (denom + 1.0)).mean()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        gt_masks=None,
        gt_boxes=None,
        orig_size=None,
        **kwargs,
    ):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "logits_to_keep": 1,
            **kwargs,
        }
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        with torch.no_grad():
            base_outputs = self.base_model(**model_inputs)
        lm_loss = getattr(base_outputs, "loss", None)

        if gt_masks is None or gt_boxes is None:
            return base_outputs

        if pixel_values is None or image_grid_thw is None:
            raise ValueError("segmentation training requires pixel_values and image_grid_thw")
        if image_grid_thw.shape[0] != gt_masks.shape[0]:
            raise ValueError("phase 1 segmentation supports exactly one image per sample")

        with torch.no_grad():
            image_outputs = self.base_model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        image_features = self._image_features_to_tensor(image_outputs)
        image_features = self._split_image_features(image_features, image_grid_thw)

        gt_masks = gt_masks.to(device=image_features.device, dtype=image_features.dtype)
        gt_boxes = gt_boxes.to(device=image_features.device, dtype=image_features.dtype)
        box_gate = self._box_gate(gt_boxes, self.seg_mask_size).to(dtype=image_features.dtype)

        pred_masks = self.mask_head(image_features, box_gate)
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        dice_loss = self._dice_loss(pred_masks, gt_masks)
        seg_loss = bce_loss + dice_loss
        loss = self.seg_loss_weight * seg_loss

        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "seg_loss": seg_loss,
            "pred_masks": pred_masks,
            "logits": base_outputs.logits,
            "past_key_values": getattr(base_outputs, "past_key_values", None),
        }

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return {f"mask_head.{k}": v for k, v in self.mask_head.state_dict(*args, **kwargs).items()}

    def load_state_dict(self, state_dict, strict=True):
        mask_state = {
            key.removeprefix("mask_head."): value
            for key, value in state_dict.items()
            if key.startswith("mask_head.")
        }
        return self.mask_head.load_state_dict(mask_state, strict=strict)

    def enable_input_require_grads(self):
        return self.base_model.enable_input_require_grads()

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()
