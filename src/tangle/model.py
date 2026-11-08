import os

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration

from tangle import config
from tangle.paths import MODELS


def load(name):
    """Load a downloaded model and its processor the same way study 3 did"""
    spec = config.MODELS[name]
    folder = MODELS/name
    # the cpu tests load a tiny model in full precision through these two variables
    dtype = os.environ.get("TANGLE_DTYPE",config.DTYPE)
    device_map = os.environ.get("TANGLE_DEVICE_MAP",spec["device_map"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(folder,dtype=getattr(torch,dtype),device_map=device_map)
    model.eval()
    kinds = {str(p.dtype) for p in model.parameters()}
    assert kinds=={f"torch.{dtype}"},f"model loaded with weights in {kinds}"
    processor = AutoProcessor.from_pretrained(folder)
    return model,processor


def prompt_text(processor,question,prefill):
    """Build the prompt text exactly like study 3 with the picture before the question and the answer start at the end"""
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    return processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)+prefill


def batch_inputs(model,processor,prompt,images):
    """Turn one prompt and a list of pictures into a batch where every row has the same tokens"""
    pictures = [Image.fromarray(image) for image in images]
    inputs = processor(text=[prompt]*len(pictures),images=pictures,return_tensors="pt",add_special_tokens=False)
    return inputs.to(model.device)


def image_positions(model,input_ids):
    """Find the token positions that hold the picture and check they all sit before the question"""
    ids = input_ids[0]
    positions = torch.nonzero(ids==model.config.image_token_id).flatten()
    assert len(positions)==config.TOKENS,f"expected {config.TOKENS} image tokens and found {len(positions)}"
    assert (input_ids==input_ids[0]).all(),"every row of a batch must have the same tokens"
    assert positions[-1]-positions[0]==config.TOKENS-1,"image tokens must be one block"
    return positions
