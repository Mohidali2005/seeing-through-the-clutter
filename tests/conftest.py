import shutil

import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from transformers import Qwen3VLConfig
from transformers import Qwen3VLForConditionalGeneration

from tangle import config
from tangle import data
from tangle.paths import ROOT

PROCESSOR = ROOT/"cache"/"processor_qwen4b"


@pytest.fixture(scope="session")
def processor_dir():
    """Download the real processor files of the pinned 4b model once"""
    spec = config.MODELS["qwen4b"]
    snapshot_download(spec["repo"],revision=spec["revision"],local_dir=str(PROCESSOR),allow_patterns=["*.json","*.txt","*.jinja","tokenizer*"])
    return PROCESSOR


@pytest.fixture(scope="session")
def processor(processor_dir):
    """The real processor of the pinned 4b model"""
    return AutoProcessor.from_pretrained(processor_dir)


@pytest.fixture(scope="session")
def tiny(processor_dir,tmp_path_factory):
    """A tiny random model with the real token ids saved next to the real processor files"""
    real = Qwen3VLConfig.from_pretrained(processor_dir)
    text = real.text_config.to_dict()
    text.update({"hidden_size":64,"intermediate_size":128,"num_hidden_layers":6,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16})
    text["rope_parameters"] = dict(text["rope_parameters"],mrope_section=[4,2,2])
    # the real configs say bfloat16 and a fresh model would build its language part in that type
    text["dtype"] = "float32"
    vision = real.vision_config.to_dict()
    vision.update({"depth":4,"hidden_size":32,"intermediate_size":64,"num_heads":2,"out_hidden_size":64,"deepstack_visual_indexes":[0,1,2],"dtype":"float32"})
    tiny_config = Qwen3VLConfig(text_config=text,vision_config=vision,image_token_id=real.image_token_id,video_token_id=real.video_token_id,
                                vision_start_token_id=real.vision_start_token_id,vision_end_token_id=real.vision_end_token_id,tie_word_embeddings=False)
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(tiny_config).eval()
    folder = tmp_path_factory.mktemp("models")/"qwen4b"
    shutil.copytree(processor_dir,folder)
    model.save_pretrained(folder)
    return model,folder.parent


@pytest.fixture(scope="session")
def scene():
    """The first objects scene with its clean and cluttered pictures"""
    first = data.load_scenes("objects",1)[0]
    return first,[data.load_image("objects",row) for row in first["rows"]]
