# weights pinned to the exact commits used in study 3 so every run sees the same model
MODELS = {
    "qwen4b":{
        "repo":"Qwen/Qwen3-VL-4B-Instruct",
        "revision":"ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "device_map":"cuda:0",
        # the 4b model fits on one t4 so each gpu runs its own copy on half the scenes
        "copies":2,
    },
    "qwen8b":{
        "repo":"Qwen/Qwen3-VL-8B-Instruct",
        "revision":"0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "device_map":"auto",
        "copies":1,
        # the replication runs the main swap rows only because the 8b model is twice as slow on one copy
        "swap_rows":"short",
    },
}

# sha of metadata.csv and of all image bytes in metadata order from the local build so the kaggle copy must match it
DATASETS = {
    "objects":{
        "repo":"atml-shapes-vqa/objects-vqa",
        "folder":"objects_vqa",
        "prefill":"apple:",
        "metadata_md5":"65431e98ba8ffef8f1adeee0658965fe",
        "images_sha256":"56ad0544bd8a10e280f72c7ee68689a3c1b880a07d0b5bf1e238b501ea5251b5",
        "swap_rows":"full",
    },
    "shapes":{
        "repo":"atml-shapes-vqa/shapes-vqa-distractors",
        "folder":"shapes_vqa_dataset",
        "prefill":"circle:",
        "metadata_md5":"1714b1274f12f0ee41fc129bdfc935a8",
        "images_sha256":"86922c45ffd3cd0a7ab59c313bd0ebe95da3c5293260cfbca3a8ec124a29b40c",
        "swap_rows":"short",
    },
    "hard":{
        "repo":"atml-shapes-vqa/shapes-vqa-hard",
        "folder":"shapes_vqa_hard",
        "prefill":"circle:",
        "metadata_md5":"7799d87facb828f86bdd46b31ecb398d",
        "images_sha256":"765ed7b2e44aac3c7ecc5154414872f9280bff1fdb00cebb3bac2d05b8d42f79",
        "swap_rows":"short",
    },
}

DTYPE = "float16"
MAX_NEW_TOKENS = 64
IMAGE_SIZE = 512
SQUARE = 32
PATCH = 16
GRID = IMAGE_SIZE//SQUARE
TOKENS = GRID*GRID
PATCHES = 4*TOKENS

# residual stream points saved for the probes where k means after k decoder layers and 0 is the hand over
# the vision points are the patch embedding and the blocks at a quarter half three quarters and the end of the vision part
LLM_NOTE_LAYERS = [3,6,8,12,18,24,30,36]

# where each image token takes its notes from in a swap row
# 0 is the clean picture and 1 the cluttered one and noise moves the cluttered notes as far as the leak in a random direction
# when is all for every layer or hand for the hand over only or a number for that one decoder layer only
SWAP = {
    "A":{"when":"all","object":1,"background":1,"clutter":1},
    "B":{"when":"all","object":0,"background":0,"clutter":1},
    "C":{"when":"all","object":1,"background":1,"clutter":0},
    "D":{"when":"all","object":0,"background":0,"clutter":0},
    "B_obj":{"when":"all","object":0,"background":1,"clutter":1},
    "B_bg":{"when":"all","object":1,"background":0,"clutter":1},
    "B_noise":{"when":"all","object":"noise","background":"noise","clutter":1},
    "B_hand":{"when":"hand","object":0,"background":0,"clutter":1},
    "C_hand":{"when":"hand","object":1,"background":1,"clutter":0},
}
SWEEP_LAYERS = [2,4,6,8,10,12,16,20,24,28]
for layer in SWEEP_LAYERS:
    SWAP[f"cut_{layer}"] = {"when":layer,"object":0,"background":0,"clutter":1}

# rows that get a real greedy answer and rows that only get the log probability of the correct answer
ANSWER_ROWS = {
    "full":["A","B","C","D","B_obj","B_bg","B_noise","B_hand","C_hand"],
    "short":["A","B","C","D","B_obj","B_bg"],
}
SCORE_ROWS = {
    "full":ANSWER_ROWS["full"]+[f"cut_{layer}" for layer in SWEEP_LAYERS],
    "short":ANSWER_ROWS["short"],
}

SEED = 0
CHUNK = 20

# the order of the full runs so a timeout keeps the most important work
RUNS = {
    "qwen4b":[("notes","objects"),("swap","objects"),("notes","shapes"),("notes","hard"),("swap","shapes"),("swap","hard")],
    "qwen8b":[("notes","objects"),("swap","objects")],
}
SMOKE_SCENES = 10

# no new step starts after this many hours and workers stop between chunks after the hard one
GEN_DEADLINE_H = 10.0
HARD_DEADLINE_H = 11.0
