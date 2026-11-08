import time

import numpy as np
import torch

from tangle import config
from tangle import data
from tangle import hooks
from tangle import model as qwen
from tangle import records
from tangle import work
from tangle.paths import NOTES


def leak(a,b):
    """One minus the cosine similarity of two sets of notes and exactly zero where they are identical"""
    out = 1-torch.nn.functional.cosine_similarity(a.float(),b.float(),dim=-1)
    return torch.where((a==b).all(-1),torch.zeros_like(out),out)


def relative(a,b):
    """Size of the change between two sets of notes relative to the size of the clean notes"""
    return (a.float()-b.float()).norm(dim=-1)/b.float().norm(dim=-1).clamp_min(1e-6)


def means(points,r,index,layers):
    """Average the notes of the chosen tokens of one row at the chosen layers"""
    return torch.stack([points[k][r][index].float().mean(0) for k in layers]).half().cpu().numpy()


if __name__=="__main__":
    args = work.worker_args("record the notes of every scene for the leak and read back tests")
    model,processor = qwen.load(args.model)
    chunks,total = work.my_chunks(args.dataset,args.worker,args.workers)
    depth = model.config.vision_config.depth
    vision_layers = [round(i*depth/4) for i in range(5)]
    llm_layers = [0]+[k for k in config.LLM_NOTE_LAYERS if k<=model.config.text_config.num_hidden_layers]
    started = time.time()
    done = 0
    for k,chunk in chunks:
        out = NOTES/args.model/args.dataset/f"chunk_{k:03d}.npz"
        if out.exists():
            continue
        if work.past_deadline():
            print(f"notes {args.model} {args.dataset}: stopping at the deadline before chunk {k}")
            break
        images_info = []
        arrays = {}
        objects = []
        dup_diff = []
        for scene in chunk:
            rows = scene["rows"]
            pictures = [data.load_image(args.dataset,row) for row in rows]
            # the clean picture goes in twice so the two copies show whether the gpu gives the same notes every time
            batch = [pictures[0],pictures[0]]+pictures[1:]
            prompt = qwen.prompt_text(processor,rows[0]["question"],config.DATASETS[args.dataset]["prefill"])
            recs = []
            pixels = []
            # one picture per pass because the place of a picture inside a batch changes the rounding of the vision part
            for picture in batch:
                inputs = qwen.batch_inputs(model,processor,prompt,[picture])
                assert inputs["image_grid_thw"].tolist()==[[1,2*config.GRID,2*config.GRID]],"the processor resized the picture"
                positions = qwen.image_positions(model,inputs["input_ids"])
                with hooks.Recorder(model,positions,1) as rec,torch.no_grad():
                    model.model(**inputs)
                recs.append(rec)
                pixels.append(inputs["pixel_values"])
            vision = [torch.cat([rec.vision[k] for rec in recs]) for k in range(len(recs[0].vision))]
            llm = [torch.cat([rec.llm[k] for rec in recs]) for k in range(len(recs[0].llm))]
            deep = [torch.cat([rec.deepstack[k] for rec in recs]) for k in range(len(recs[0].deepstack))]
            assert len(llm)==model.config.text_config.num_hidden_layers+1,"missed a residual point"
            dup_diff.append(max(float((p[1].float()-p[0].float()).abs().max()) for p in vision+llm+deep))
            pixels = torch.stack(pixels).reshape(len(batch),config.PATCHES,-1)
            for i,row in enumerate(rows[1:]):
                r = i+2
                m = data.masks(pictures[0],pictures[i+1],data.objects_of(row))
                same_pixels = (pixels[r]==pixels[0]).all(-1).cpu().numpy()
                record = {
                    "untouched":m["untouched"],"object":m["object"],"background":m["background"],"clutter":m["clutter"],
                    "untouched_patch":m["untouched_patch"],"object_patch":m["object_patch"],"background_patch":m["background_patch"],
                    "leak_vision":torch.stack([leak(p[r],p[0]) for p in vision]).cpu().numpy(),
                    "rel_vision":torch.stack([relative(p[r],p[0]) for p in vision]).half().cpu().numpy(),
                    "leak_llm":torch.stack([leak(p[r],p[0]) for p in llm]).cpu().numpy(),
                    "rel_llm":torch.stack([relative(p[r],p[0]) for p in llm]).half().cpu().numpy(),
                    "leak_deep":torch.stack([leak(p[r],p[0]) for p in deep]).cpu().numpy(),
                    "rel_deep":torch.stack([relative(p[r],p[0]) for p in deep]).half().cpu().numpy(),
                    # both must be zero when the squares line up with the pixels
                    "pixel_mismatch":np.int64((same_pixels!=m["untouched_patch"]).sum()),
                    "embed_diff":np.int64(((vision[0][r]!=vision[0][0]).any(-1).cpu().numpy()&m["untouched_patch"]).sum()),
                    "nonfinite":np.int64(sum(int((~torch.isfinite(p[r])).sum()) for p in vision+llm+deep)),
                }
                bg_sq = np.flatnonzero(m["background"])
                bg_patch = np.flatnonzero(m["background_patch"])
                record["bg_vision_clean"] = means(vision,0,bg_patch,vision_layers)
                record["bg_vision_clut"] = means(vision,r,bg_patch,vision_layers)
                record["bg_llm_clean"] = means(llm,0,bg_sq,llm_layers)
                record["bg_llm_clut"] = means(llm,r,bg_sq,llm_layers)
                for name,value in record.items():
                    arrays.setdefault(name,[]).append(value)
                for t in m["targets"]:
                    if len(t["squares"])==0 or len(t["patches"])==0:
                        continue
                    objects.append({
                        "obj_image":len(images_info),"obj_id":t["id"],"obj_shape":t["shape"],
                        "obj_squares":len(t["squares"]),"obj_patches":len(t["patches"]),
                        "obj_vision_clean":means(vision,0,t["patches"],vision_layers),
                        "obj_vision_clut":means(vision,r,t["patches"],vision_layers),
                        "obj_llm_clean":means(llm,0,t["squares"],llm_layers),
                        "obj_llm_clut":means(llm,r,t["squares"],llm_layers),
                    })
                images_info.append({"image_id":row["image_id"],"scene_id":int(row["scene_id"]),"version":row["version"],"split":row["split"]})
        result = {name:np.stack(values) for name,values in arrays.items()}
        for name in ["image_id","scene_id","version","split"]:
            result[name] = np.array([info[name] for info in images_info])
        for name in objects[0]:
            result[name] = np.stack([o[name] for o in objects]) if isinstance(objects[0][name],np.ndarray) else np.array([o[name] for o in objects])
        result["dup_diff"] = np.array(dup_diff)
        result["vision_layers"] = np.array(vision_layers)
        result["llm_layers"] = np.array(llm_layers)
        records.write_npz(out,result)
        done += 1
    print(f"notes {args.model} {args.dataset} worker {args.worker}: {done} new chunks of {total} in {(time.time()-started)/60:.1f} min")
