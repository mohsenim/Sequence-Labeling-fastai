from fastai.text.all import *


def _get_tokenizer(ds):
    tok = getattr(ds, "tokenizer", None)
    if isinstance(tok, Tokenizer):
        return tok
    if isinstance(tok, (list, L)):
        for t in tok:
            if isinstance(t, Tokenizer):
                return t


def _get_lengths(ds):
    tok = _get_tokenizer(ds)
    if tok is None:
        return
    return tok.get_lengths(ds.items)


def _maybe_first_or_second(o, i=0):
    return o[i] if isinstance(o, tuple) else o


@delegates()
class SequentialDataLoader(TfmdDL):
    "A `DataLoader` suitable for language modeling"

    def __init__(
        self, dataset, lens=None, cache=2, bs=64, seq_len=72, num_workers=0, **kwargs
    ):
        self.items = ReindexCollection(
            dataset, cache=cache, tfm=partial(_maybe_first_or_second, i=0)
        )
        self.items_y = ReindexCollection(
            dataset, cache=cache, tfm=partial(_maybe_first_or_second, i=1)
        )
        self.seq_len = seq_len
        if lens is None:
            lens = _get_lengths(dataset)
        if lens is None:
            lens = [len(o) for o in self.items]
        self.lens = ReindexCollection(lens, idxs=self.items.idxs)
        # The "-1" is to allow for final label, we throw away the end that's less than bs
        corpus = round_multiple(sum(lens) - 1, bs, round_down=True)
        self.bl = corpus // bs  # bl stands for batch length
        self.n_batches = self.bl // (seq_len) + int(self.bl % seq_len != 0)
        self.last_len = self.bl - (self.n_batches - 1) * seq_len
        self.make_chunks()
        super().__init__(dataset=dataset, bs=bs, num_workers=num_workers, **kwargs)
        self.n = self.n_batches * bs

    def make_chunks(self):
        self.chunks = Chunks(self.items, self.lens)
        self.chunks_y = Chunks(self.items_y, self.lens)

    def shuffle_fn(self, idxs):
        self.items.shuffle()
        self.items_y.reindex(self.items.idxs)
        self.make_chunks()
        return idxs

    def create_item(self, seq):
        if seq is None:
            seq = 0
        if seq >= self.n:
            raise IndexError
        sl = self.last_len if seq // self.bs == self.n_batches - 1 else self.seq_len
        st = (seq % self.bs) * self.bl + (seq // self.bs) * self.seq_len
        txt = self.chunks[st : st + sl + 1]
        y = self.chunks_y[st : st + sl + 1]
        return LMTensorText(txt), y

    @delegates(TfmdDL.new)
    def new(self, dataset=None, seq_len=None, **kwargs):
        lens = self.lens.coll if dataset is None else None
        seq_len = self.seq_len if seq_len is None else seq_len
        return super().new(dataset=dataset, lens=lens, seq_len=seq_len, **kwargs)


class SequentialTextBlock(TransformBlock):
    "A `TransformBlock` for texts"

    @delegates(Numericalize.__init__)
    def __init__(self, tok_tfm, vocab=None, seq_len=72, **kwargs):
        type_tfms = [tok_tfm, Numericalize(vocab, **kwargs)]
        return super().__init__(
            type_tfms=type_tfms,
            dl_type=SequentialDataLoader,
            dls_kwargs={"seq_len": seq_len},
        )

    @classmethod
    @delegates(Tokenizer.from_df, keep=True)
    def from_df(
        cls, text_cols, vocab=None, seq_len=72, bmin_freq=3, max_vocab=60000, **kwargs
    ):
        "Build a `TextBlock` from a dataframe using `text_cols`"
        return cls(
            Tokenizer.from_df(text_cols, **kwargs),
            vocab=vocab,
            seq_len=seq_len,
            min_freq=min_freq,
            max_vocab=max_vocab,
        )

    @classmethod
    @delegates(Tokenizer.from_folder, keep=True)
    def from_folder(
        cls, path, vocab=None, seq_len=72, min_freq=3, max_vocab=60000, **kwargs
    ):
        "Build a `TextBlock` from a `path`"
        return cls(
            Tokenizer.from_folder(path, **kwargs),
            vocab=vocab,
            seq_len=seq_len,
            min_freq=min_freq,
            max_vocab=max_vocab,
        )
