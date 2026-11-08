import torch

from tangle import config


def parts_of(model):
    """Return the vision part and the language part of the model"""
    return model.model.visual,model.model.language_model


def hidden_of(args,kwargs):
    """Take the hidden states out of the arguments of a decoder layer"""
    return args[0] if args else kwargs["hidden_states"]


def with_hidden(args,kwargs,hidden):
    """Put new hidden states back into the arguments of a decoder layer"""
    if args:
        return (hidden,*args[1:]),kwargs
    return args,dict(kwargs,hidden_states=hidden)


class Recorder:
    """Keep the notes of every patch and image token at every layer for all rows of one batch"""

    def __init__(self,model,positions,rows):
        self.model = model
        self.positions = positions
        self.rows = rows
        self.vision = []
        self.deepstack = []
        self.llm = []
        self.handles = []

    def keep_vision(self,module,args,output):
        """Keep the output of the patch embedding or of one vision block"""
        self.vision.append(output.reshape(self.rows,config.PATCHES,-1).clone())

    def keep_handover(self,module,args,output):
        """Keep the three deepstack feature sets handed to the language part"""
        self.deepstack = [d.reshape(self.rows,config.TOKENS,-1).clone() for d in output.deepstack_features]

    def keep_llm(self,module,args,kwargs=None):
        """Keep the residual stream at the image tokens as it enters a decoder layer or the final norm"""
        hidden = hidden_of(args,kwargs or {})
        if hidden.shape[1]>1:
            self.llm.append(hidden[:,self.positions.to(hidden.device)].clone())

    def __enter__(self):
        visual,language = parts_of(self.model)
        self.handles.append(visual.patch_embed.register_forward_hook(self.keep_vision))
        for block in visual.blocks:
            self.handles.append(block.register_forward_hook(self.keep_vision))
        self.handles.append(visual.register_forward_hook(self.keep_handover))
        for layer in language.layers:
            self.handles.append(layer.register_forward_pre_hook(self.keep_llm,with_kwargs=True))
        self.handles.append(language.norm.register_forward_pre_hook(self.keep_llm))
        return self

    def __exit__(self,*exc):
        for handle in self.handles:
            handle.remove()


class Surgery:
    """Give every mixed row of a batch notes taken from the clean row 0 and the cluttered row 1"""

    def __init__(self,model,positions,plans,parts,seed):
        self.model = model
        self.positions = positions
        self.plans = plans
        self.seed = seed
        self.handles = []
        # 0 means clean and 1 cluttered and 2 the cluttered notes moved in a random direction
        code = {0:0,1:1,"noise":2}
        self.sources = {}
        for r,plan in enumerate(plans):
            if plan is None:
                continue
            assert r>1,"rows 0 and 1 must stay plain"
            source = torch.empty(config.TOKENS,dtype=torch.long)
            for part in ["object","background","clutter"]:
                source[torch.as_tensor(parts[part])] = code[plan[part]]
            self.sources[r] = source[:,None]

    def rows_for(self,when):
        """List the rows whose notes are replaced at this point"""
        rows = []
        for r,plan in enumerate(self.plans):
            if plan is None:
                continue
            if plan["when"]==when or (plan["when"]=="all" and when!="hand"):
                rows.append(r)
            # a row cut at one layer copies the cluttered notes before that layer so it carries no noise of its own
            elif isinstance(plan["when"],int) and when!="hand" and when<plan["when"]:
                rows.append(r)
        return rows

    def mix(self,r,clean,cluttered,layer):
        """Build the notes of one mixed row from the clean and cluttered notes"""
        source = self.sources[r].to(clean.device)
        if isinstance(self.plans[r]["when"],int) and layer<self.plans[r]["when"]:
            return cluttered
        out = torch.where(source==0,clean,cluttered)
        if (source==2).any():
            generator = torch.Generator(device=clean.device).manual_seed(self.seed*1000+layer)
            direction = torch.randn(clean.shape,generator=generator,device=clean.device,dtype=torch.float32)
            direction = direction/direction.norm(dim=-1,keepdim=True)
            size = (clean.float()-cluttered.float()).norm(dim=-1,keepdim=True)
            moved = (cluttered.float()+direction*size).to(clean.dtype)
            out = torch.where(source==2,moved,out)
        return out

    def before_layer(self,layer):
        """Make the hook that replaces image notes as they enter one decoder layer"""
        def hook(module,args,kwargs):
            hidden = hidden_of(args,kwargs)
            rows = self.rows_for(layer)
            # only the first pass over the prompt carries the image tokens
            if hidden.shape[1]==1 or not rows:
                return None
            positions = self.positions.to(hidden.device)
            hidden = hidden.clone()
            clean = hidden[0,positions]
            cluttered = hidden[1,positions]
            for r in rows:
                hidden[r,positions] = self.mix(r,clean,cluttered,layer)
            return with_hidden(args,kwargs,hidden)
        return hook

    def at_handover(self,module,args,output):
        """Replace the merger output and the deepstack features of the hand over rows"""
        rows = self.rows_for("hand")
        if not rows:
            return output
        for features in [output.pooler_output,*output.deepstack_features]:
            view = features.view(len(self.plans),config.TOKENS,-1)
            clean = view[0].clone()
            cluttered = view[1].clone()
            for r in rows:
                view[r] = self.mix(r,clean,cluttered,-1)
        return output

    def __enter__(self):
        visual,language = parts_of(self.model)
        self.handles.append(visual.register_forward_hook(self.at_handover))
        for i,layer in enumerate(language.layers):
            self.handles.append(layer.register_forward_pre_hook(self.before_layer(i),with_kwargs=True))
        return self

    def __exit__(self,*exc):
        for handle in self.handles:
            handle.remove()
