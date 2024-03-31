from fastai.text.all import *


class SequentialCategorize(Categorize):
    "Reversible transform of sequential-category strings to `vocab` id"
    loss_func, order = BCEWithLogitsLossFlat(), 1

    def __init__(self, vocab=None, add_na=False):
        super().__init__(vocab=vocab, add_na=add_na, sort=vocab == None)

    def setups(self, dsets):
        if not dsets:
            return
        if self.vocab is None:
            vals = set()
            for b in dsets:
                vals = vals.union(set(b))
            self.vocab = CategoryMap(list(vals), add_na=self.add_na)

    def encodes(self, o):
        if not all(elem in self.vocab.o2i.keys() for elem in o):
            diff = [elem for elem in o if elem not in self.vocab.o2i.keys()]
            diff_str = "', '".join(diff)
            raise KeyError(
                f"Labels '{diff_str}' were not included in the training dataset"
            )
        return TensorMultiCategory([self.vocab.o2i[o_] for o_ in o])

    def decodes(self, o):
        return SequentialCategory([self.vocab[o_] for o_ in o])


class SequentialCategory(L):
    def show(self, ctx=None, sep=";", color="black", **kwargs):
        return show_title(sep.join(self.map(str)), ctx=ctx, color=color, **kwargs)


def SequentialCategoryBlock(
    vocab: MutableSequence | pd.Series = None,  # List of unique class names
    add_na: bool = False,  # Add `#na#` to `vocab`
):
    "`TransformBlock` for sequential categorical targets"
    tfm = [SequentialCategorize(vocab=vocab, add_na=add_na)]
    return TransformBlock(type_tfms=tfm)
