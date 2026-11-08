import numpy as np
import torch

from tangle import config
from tangle import data
from tangle import hooks
from tangle import model as qwen


def test_patch_order_matches_processor(processor):
    """Changing one pixel must change exactly the patch our map puts there"""
    rng = np.random.default_rng(0)
    clean = rng.integers(0,256,(512,512,3),dtype=np.uint8)
    cells = data.patch_cells()
    for _ in range(8):
        y,x = rng.integers(0,512,2)
        changed = clean.copy()
        changed[y,x] = 255-changed[y,x]
        out = processor.image_processor(images=[clean,changed],return_tensors="pt")
        assert out["image_grid_thw"].tolist()==[[1,32,32],[1,32,32]]
        pixels = out["pixel_values"].reshape(2,config.PATCHES,-1)
        differ = torch.nonzero((pixels[0]!=pixels[1]).any(-1)).flatten().tolist()
        assert len(differ)==1
        assert tuple(cells[differ[0]])==(y//16,x//16)


def test_masks_match_processed_pixels(processor):
    """Untouched patches from the png must be exactly the patches the processor leaves unchanged"""
    for first in data.load_scenes("objects",3):
        clean = data.load_image("objects",first["rows"][0])
        for row in first["rows"][1:]:
            image = data.load_image("objects",row)
            m = data.masks(clean,image,data.objects_of(row))
            out = processor.image_processor(images=[clean,image],return_tensors="pt")
            pixels = out["pixel_values"].reshape(2,config.PATCHES,-1)
            same = (pixels[0]==pixels[1]).all(-1).numpy()
            assert (same==m["untouched_patch"]).all()
            assert m["object"].sum()>0 and m["clutter"].sum()>0
            assert not (m["object"]&m["clutter"]).any()
            assert (m["object"]|m["background"]|m["clutter"]).all()
            # a square is untouched only when all four of its patches are
            assert (m["untouched_patch"].reshape(config.TOKENS,4).all(1)==m["untouched"]).all()


def test_prompt_puts_the_picture_first(processor,tiny,scene):
    """The prompt must match study 3 with the picture before the question and the answer start last"""
    model,_ = tiny
    first,pictures = scene
    prompt = qwen.prompt_text(processor,first["rows"][0]["question"],"apple:")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\napple:") or prompt.endswith("<|im_start|>assistant\napple:")
    assert prompt.index("<|vision_start|>")<prompt.index("Count the objects")
    inputs = qwen.batch_inputs(model,processor,prompt,pictures[:2])
    positions = qwen.image_positions(model,inputs["input_ids"])
    assert positions[-1]<inputs["input_ids"].shape[1]-50


def test_recorder_is_exact(processor,tiny,scene):
    """Two copies of the clean picture give identical notes and untouched patches start out identical"""
    model,_ = tiny
    first,pictures = scene
    prompt = qwen.prompt_text(processor,first["rows"][0]["question"],"apple:")
    recs = []
    for picture in [pictures[0],pictures[0],pictures[2]]:
        inputs = qwen.batch_inputs(model,processor,prompt,[picture])
        positions = qwen.image_positions(model,inputs["input_ids"])
        with hooks.Recorder(model,positions,1) as rec,torch.no_grad():
            model.model(**inputs)
        recs.append(rec)
    clean,again,cluttered = recs
    assert len(clean.vision)==model.config.vision_config.depth+1
    assert len(clean.llm)==model.config.text_config.num_hidden_layers+1
    assert len(clean.deepstack)==3
    for a,b in zip(clean.vision+clean.llm+clean.deepstack,again.vision+again.llm+again.deepstack):
        assert torch.equal(a,b)
    m = data.masks(pictures[0],pictures[2],data.objects_of(first["rows"][2]))
    untouched = torch.as_tensor(m["untouched_patch"])
    assert torch.equal(cluttered.vision[0][0][untouched],clean.vision[0][0][untouched])
    # full attention spreads the clutter so untouched squares change later on
    squares = torch.as_tensor(m["untouched"])
    assert not torch.equal(cluttered.llm[-1][0][squares],clean.llm[-1][0][squares])


def test_swap_rows_take_the_right_notes(processor,tiny,scene):
    """Every swap row gets clean or cluttered notes exactly where its plan says at exactly the planned layers"""
    model,_ = tiny
    first,pictures = scene
    row = first["rows"][2]
    m = data.masks(pictures[0],pictures[2],data.objects_of(row))
    parts = {"object":m["object"],"background":m["background"],"clutter":m["clutter"]}
    names = ["clean","cluttered","A","B","C","D","B_obj","B_bg","B_noise","B_hand","C_hand","cut_2"]
    plans = [None,None]+[config.SWAP[n] for n in names[2:]]
    prompt = qwen.prompt_text(processor,first["rows"][0]["question"],"apple:")
    inputs = qwen.batch_inputs(model,processor,prompt,[pictures[0]]+[pictures[2]]*(len(names)-1))
    positions = qwen.image_positions(model,inputs["input_ids"])
    with hooks.Surgery(model,positions,plans,parts,seed=7),hooks.Recorder(model,positions,len(names)) as rec,torch.no_grad():
        logits = model(**inputs,logits_to_keep=4).logits
    r = {n:i for i,n in enumerate(names)}
    obj = torch.as_tensor(m["object"])
    bg = torch.as_tensor(m["background"])
    clut = torch.as_tensor(m["clutter"])
    untouched = obj|bg
    layers = model.config.text_config.num_hidden_layers
    for k in range(layers):
        h = rec.llm[k]
        assert torch.equal(h[r["A"]],h[1])
        assert torch.equal(h[r["D"]],h[0])
        assert torch.equal(h[r["B"]][untouched],h[0][untouched]) and torch.equal(h[r["B"]][clut],h[1][clut])
        assert torch.equal(h[r["C"]][untouched],h[1][untouched]) and torch.equal(h[r["C"]][clut],h[0][clut])
        assert torch.equal(h[r["B_obj"]][obj],h[0][obj]) and torch.equal(h[r["B_obj"]][bg|clut],h[1][bg|clut])
        assert torch.equal(h[r["B_bg"]][bg],h[0][bg]) and torch.equal(h[r["B_bg"]][obj|clut],h[1][obj|clut])
        moved = (h[r["B_noise"]][untouched]-h[1][untouched]).norm(dim=-1)
        leak = (h[0][untouched]-h[1][untouched]).norm(dim=-1)
        assert torch.allclose(moved,leak,rtol=1e-4,atol=1e-5) and torch.equal(h[r["B_noise"]][clut],h[1][clut])
        if k>0:
            assert not torch.equal(h[r["B_noise"]][untouched],h[0][untouched])
    hand = rec.llm[0]
    assert torch.equal(hand[r["B_hand"]][untouched],hand[0][untouched]) and torch.equal(hand[r["B_hand"]][clut],hand[1][clut])
    assert torch.equal(hand[r["C_hand"]][untouched],hand[1][untouched]) and torch.equal(hand[r["C_hand"]][clut],hand[0][clut])
    for deep in rec.deepstack:
        assert torch.equal(deep[r["B_hand"]][untouched],deep[0][untouched]) and torch.equal(deep[r["B_hand"]][clut],deep[1][clut])
    # after the hand over the language part recomputes these rows so the leak can come back
    assert not torch.equal(rec.llm[3][r["B_hand"]][untouched],rec.llm[3][0][untouched])
    assert torch.equal(rec.llm[0][r["cut_2"]],rec.llm[0][1]) and torch.equal(rec.llm[1][r["cut_2"]],rec.llm[1][1])
    assert torch.equal(rec.llm[2][r["cut_2"]][untouched],rec.llm[2][0][untouched])
    assert torch.equal(logits[r["A"]],logits[1])
    assert torch.equal(logits[r["D"]],logits[0])
    assert not torch.equal(logits[r["B"]],logits[r["A"]])


def test_generation_keeps_identities(processor,tiny,scene):
    """Greedy answers of row A equal the plain cluttered row and row D the plain clean row"""
    model,_ = tiny
    first,pictures = scene
    row = first["rows"][3]
    m = data.masks(pictures[0],pictures[3],data.objects_of(row))
    parts = {"object":m["object"],"background":m["background"],"clutter":m["clutter"]}
    names = ["clean","cluttered","A","D","B"]
    plans = [None,None]+[config.SWAP[n] for n in names[2:]]
    prompt = qwen.prompt_text(processor,first["rows"][0]["question"],"apple:")
    inputs = qwen.batch_inputs(model,processor,prompt,[pictures[0]]+[pictures[3]]*4)
    positions = qwen.image_positions(model,inputs["input_ids"])
    with hooks.Surgery(model,positions,plans,parts,seed=1),torch.no_grad():
        out = model.generate(**inputs,max_new_tokens=12,do_sample=False)
    assert torch.equal(out[2],out[1])
    assert torch.equal(out[3],out[0])
    with torch.no_grad():
        alone = model.generate(**qwen.batch_inputs(model,processor,prompt,[pictures[3]]),max_new_tokens=12,do_sample=False)
    assert torch.equal(alone[0],out[1])
