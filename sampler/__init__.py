from sampler.modelco import ModelCoSampler
from sampler.exomild import ExoMILDSampler, ExoMILDEnsemblingSampler
from sampler.unet import UnetSampler


def get_sampler(name, mode, model, **kwargs):
    if name == "modelco":
        return ModelCoSampler(
            model=model,
            **kwargs,
        )
    elif name == "exomild":
        if mode == "train":
            return ExoMILDSampler(
                model=model,
                **kwargs,
            )
        # elif name == "exomild_ensembling":
        return ExoMILDEnsemblingSampler(
            model=model,
            **kwargs,
        )
    elif name == "unet":
        return UnetSampler(model=model, **kwargs)
    else:
        raise ValueError(f"Sampler not recognized: {name}")
