from fastai.text.all import *
from core_ext import *

@delegates(Learner.__init__)
def sequential_model_learner(dls, arch, config=None, drop_mult=1., backwards=False, pretrained=True, pretrained_fnames=None, **kwargs):
    "Create a `Learner` with a language model from `dls` and `arch`."
    vocab = dls.vocab[0]
    vocab_labels = dls.vocab[1]
    model = get_sequential_model(arch, len(vocab), len(vocab_labels), config=config, drop_mult=drop_mult)
    meta = core._model_meta[arch]
    learn = LMLearner(dls, model, loss_func=CrossEntropyLossFlat(), splitter=meta['split_lm'], **kwargs)
    url = 'url_bwd' if backwards else 'url'
    if pretrained or pretrained_fnames:
        if pretrained_fnames is not None:
            fnames = [learn.path/learn.model_dir/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]
        else:
            if url not in meta:
                warn("There are no pretrained weights for that architecture yet!")
                return learn
            model_path = untar_data(meta[url] , c_key='model')
            try: fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
            except IndexError: print(f'The model in {model_path} is incomplete, download again'); raise
        learn = learn.load_pretrained(*fnames, model=learn.model[0])
        learn.freeze()
    return learn
