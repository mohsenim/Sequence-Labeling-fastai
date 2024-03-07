from pathlib import Path
from transforms_ext import *
from data_ext import SequentialTextBlock
import learner_ext
from fastai.text.all import *

def get_annotations(o):
    with open(path/'../labels'/o.name) as f:
        labels = f.read()
    return labels.split()


if __name__ == '__main__':

    path = Path('./corpus/texts')
    files = get_text_files(path)

    print(f'Number of text files: {len(files)}')

    dls = DataBlock(
        blocks=[SequentialTextBlock.from_folder(path, tok=BaseTokenizer(split_char=None), rules=[]), SequentialCategoryBlock],
        get_items=get_text_files,
        get_y=lambda o: get_annotations(o),
        splitter=RandomSplitter()
    ).dataloaders(path, path=path, seq_len=256, bs=64)

    xb, yb = dls.one_batch()
    print(xb.shape, yb.shape)

    learn = learner_ext.sequential_model_learner(
        dls,
        AWD_LSTM,
        drop_mult=0.3,
        metrics=accuracy
    )

    learn.fit_one_cycle(1, 2e-3)
    learn.unfreeze()
    learn.fit_one_cycle(10, 5e-3, cbs=SaveModelCallback(monitor='accuracy',
                                                        every_epoch=True,
                                                        fname='dialogue_model'))
