import time

import torch
import vqa_scoring

from tangle import config
from tangle import data
from tangle import hooks
from tangle import model as qwen
from tangle import records
from tangle import work
from tangle.paths import SWAPS


def plans_for(names):
    """Turn row names into swap plans where the clean and cluttered rows stay plain"""
    return [None,None]+[config.SWAP[n] for n in names[2:]]


if __name__=="__main__":
    args = work.worker_args("answer every cluttered picture with mixed clean and cluttered notes")
    model,processor = qwen.load(args.model)
    kind = config.DATASETS[args.dataset]["swap_rows"]
    prefill = config.DATASETS[args.dataset]["prefill"]
    answer_names = ["clean","cluttered"]+config.ANSWER_ROWS[kind]
    score_names = ["clean","cluttered"]+config.SCORE_ROWS[kind]
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    chunks,total = work.my_chunks(args.dataset,args.worker,args.workers)
    started = time.time()
    done = 0
    for k,chunk in chunks:
        out = SWAPS/args.model/args.dataset/f"chunk_{k:03d}.csv"
        if out.exists():
            continue
        if work.past_deadline():
            print(f"swap {args.model} {args.dataset}: stopping at the deadline before chunk {k}")
            break
        results = []
        for scene in chunk:
            clean_row = scene["rows"][0]
            clean = data.load_image(args.dataset,clean_row)
            prompt = qwen.prompt_text(processor,clean_row["question"],prefill)
            for v,row in enumerate(scene["rows"][1:]):
                image = data.load_image(args.dataset,row)
                m = data.masks(clean,image,data.objects_of(row))
                parts = {"object":m["object"],"background":m["background"],"clutter":m["clutter"]}
                seed = scene["scene_id"]*100+v
                truth = vqa_scoring.counts_from_row(row)
                assert row["answer"].startswith(prefill),"the answer must start with the prefill"
                continuation = processor.tokenizer(row["answer"][len(prefill):],add_special_tokens=False).input_ids+[end_id]

                inputs = qwen.batch_inputs(model,processor,prompt,[clean]+[image]*(len(answer_names)-1))
                positions = qwen.image_positions(model,inputs["input_ids"])
                length = inputs["input_ids"].shape[1]
                with hooks.Surgery(model,positions,plans_for(answer_names),parts,seed),torch.no_grad():
                    output = model.generate(**inputs,max_new_tokens=config.MAX_NEW_TOKENS,do_sample=False)
                texts = processor.batch_decode(output[:,length:],skip_special_tokens=True)
                new_tokens = (output[:,length:]!=processor.tokenizer.pad_token_id).sum(1).tolist()
                answers = {name:prefill+text.rstrip() for name,text in zip(answer_names,texts)}
                tokens = dict(zip(answer_names,new_tokens))

                inputs = qwen.batch_inputs(model,processor,prompt,[clean]+[image]*(len(score_names)-1))
                n = len(score_names)
                target = torch.tensor(continuation,device=model.device)
                inputs["input_ids"] = torch.cat([inputs["input_ids"],target[None].repeat(n,1)],1)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"],torch.ones(n,len(continuation),dtype=inputs["attention_mask"].dtype,device=model.device)],1)
                if "mm_token_type_ids" in inputs:
                    inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"],torch.zeros(n,len(continuation),dtype=inputs["mm_token_type_ids"].dtype,device=model.device)],1)
                with hooks.Surgery(model,positions,plans_for(score_names),parts,seed),torch.no_grad():
                    # keeping only the positions that predict the answer tokens so the logits stay small
                    logits = model(**inputs,use_cache=False,logits_to_keep=len(continuation)+1).logits[:,:-1].float()
                logp = torch.log_softmax(logits,-1)
                gold = logp.gather(-1,target[None,:,None].repeat(n,1,1).to(logp.device)).squeeze(-1).sum(-1).tolist()
                greedy_gold = (logp.argmax(-1)==target.to(logp.device)).all(-1).tolist()
                checks = {
                    "A":float((logits[score_names.index("A")]-logits[1]).abs().max()),
                    "D":float((logits[score_names.index("D")]-logits[0]).abs().max()),
                }
                for i,name in enumerate(score_names):
                    scored = vqa_scoring.score_answer(answers[name],truth) if name in answers else {}
                    results.append({
                        "model":args.model,"dataset":args.dataset,"image_id":row["image_id"],"scene_id":scene["scene_id"],
                        "version":row["version"],"split":row["split"],"condition":name,
                        "answer":answers.get(name,""),"parsed":scored.get("parsed",""),"exact_match":scored.get("exact_match",""),
                        "abs_error":scored.get("abs_error",""),"total_bias":scored.get("total_bias",""),"new_tokens":tokens.get(name,""),
                        "logprob":round(gold[i],5),"greedy_gold":greedy_gold[i],"check_diff":checks.get(name,""),
                        "n_object":int(m["object"].sum()),"n_background":int(m["background"].sum()),"n_clutter":int(m["clutter"].sum()),
                    })
        records.write_csv(out,results)
        done += 1
    print(f"swap {args.model} {args.dataset} worker {args.worker}: {done} new chunks of {total} in {(time.time()-started)/60:.1f} min")
