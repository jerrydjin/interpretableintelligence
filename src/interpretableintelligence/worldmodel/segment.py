"""Arm masks from a text-prompted segmenter (CLIPSeg), for labelling which patches show the robot.

CLIPSeg (``CIDAS/clipseg-rd64-refined``, 150M parameters, Apache 2.0) scores every pixel against a
text prompt. A pixel is arm when any prompt's probability passes ``threshold``. The masks only have
to be right at the resolution of a 16-pixel patch; ``ii action-effects`` writes overlays so a person
can check them before trusting the numbers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

CLIPSEG_ID = "CIDAS/clipseg-rd64-refined"
ARM_PROMPTS = ("a robot arm", "a robot gripper")
THRESHOLD = 0.4
_SIZE = 352
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipSegArm:
    def __init__(self, device: torch.device | str = "cpu", prompts=ARM_PROMPTS, threshold: float = THRESHOLD) -> None:
        from transformers import AutoTokenizer, CLIPSegForImageSegmentation

        self.device = torch.device(device)
        self.prompts = list(prompts)
        self.threshold = float(threshold)
        self.tokenizer = AutoTokenizer.from_pretrained(CLIPSEG_ID)
        self.model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, frames: np.ndarray, batch: int = 16) -> np.ndarray:
        """``[N, H, W, 3]`` uint8 to ``[N, H, W]`` bool arm masks."""
        n, h, w, _ = frames.shape
        text = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(self.device)
        k = len(self.prompts)
        mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)
        out = []
        for i in range(0, n, batch):
            x = torch.from_numpy(np.ascontiguousarray(frames[i : i + batch])).to(self.device).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=(_SIZE, _SIZE), mode="bilinear", align_corners=False, antialias=True)
            x = ((x - mean) / std).repeat_interleave(k, dim=0)
            ids = text["input_ids"].repeat(len(x) // k, 1)
            att = text["attention_mask"].repeat(len(x) // k, 1)
            logits = self.model(input_ids=ids, attention_mask=att, pixel_values=x).logits
            logits = logits.view(-1, k, _SIZE, _SIZE)
            prob = torch.sigmoid(logits).amax(dim=1, keepdim=True)
            prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
            out.append((prob[:, 0] > self.threshold).cpu().numpy())
        return np.concatenate(out)


GDINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam-vit-base"
GDINO_TEXT = "a robot arm. a robot gripper. a robot."


class GroundedSamArm:
    """Arm masks from a text-prompted box detector (Grounding DINO) refined by SAM inside each box.

    Stronger than CLIPSeg on DROID's small 320 x 180 frames, where CLIPSeg mistakes the arm for
    background. A box that holds an object can pull the held object into the arm mask, so ``moved``
    patches lean towards objects the arm pushes rather than carries. Needs Pillow and torchvision
    (transformers' image processors), which the Modal image has.
    """

    def __init__(self, device: torch.device | str = "cpu", text: str = GDINO_TEXT, box_threshold: float = 0.25) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.device = torch.device(device)
        self.text = text
        self.box_threshold = float(box_threshold)
        self.det_proc = AutoProcessor.from_pretrained(GDINO_ID)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(self.device).eval()
        self.sam_proc = SamProcessor.from_pretrained(SAM_ID)
        self.sam = SamModel.from_pretrained(SAM_ID).to(self.device).eval()

    @torch.inference_mode()
    def boxes(self, frames: np.ndarray, batch: int = 2) -> list[np.ndarray]:
        out = []
        h, w = frames.shape[1:3]
        for i in range(0, len(frames), batch):
            imgs = [f for f in frames[i : i + batch]]
            inputs = self.det_proc(images=imgs, text=[self.text] * len(imgs), return_tensors="pt").to(self.device)
            res = self.det_proc.post_process_grounded_object_detection(
                self.det(**inputs),
                inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=0.25,
                target_sizes=[(h, w)] * len(imgs),
            )
            out.extend(r["boxes"].float().cpu().numpy() for r in res)
        return out

    @torch.inference_mode()
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """``[N, H, W, 3]`` uint8 to ``[N, H, W]`` bool arm masks (empty where no arm box is found)."""
        n, h, w, _ = frames.shape
        masks = np.zeros((n, h, w), dtype=bool)
        for i, boxes in enumerate(self.boxes(frames)):
            if len(boxes) == 0:
                continue
            inputs = self.sam_proc(images=frames[i], input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
            pred = self.sam(**inputs, multimask_output=True).pred_masks
            m = self.sam_proc.image_processor.post_process_masks(
                pred.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )[0]  # [boxes, 3, H, W]: SAM's three nested guesses per box
            # The largest guess is the whole arm; the smaller ones tend to be one link or the gripper.
            m = m.reshape(m.shape[0], -1, h, w)
            biggest = m.flatten(2).sum(-1).argmax(dim=1)
            masks[i] = m[torch.arange(m.shape[0]), biggest].any(dim=0).numpy()
        return masks
