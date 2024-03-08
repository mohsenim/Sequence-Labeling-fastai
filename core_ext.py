from fastai.text.all import *


class LinearDecoder(Module):
    "To go on top of a RNNCore module and create a Sequential Model."
    initrange = 0.1

    def __init__(
        self,
        n_out: int,  # Number of output channels
        n_hid: int,  # Number of features in encoder last layer output
        output_p: float = 0.1,  # Input dropout probability
        bias: bool = True,  # If `False` the layer will not learn additive bias
    ):
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if bias:
            self.decoder.bias.data.zero_()

    def forward(self, input):
        dp_inp = self.output_dp(input)
        return self.decoder(dp_inp), input, dp_inp


def get_sequential_model(
    arch,  # Function or class that can generate a language model architecture
    vocab_sz: int,  # Size of the vocabulary
    output_sz: int,  # Number of labels
    config: dict = None,  # Model configuration dictionary
    drop_mult: float = 1.0,  # Multiplicative factor to scale all dropout probabilities in `config`
) -> SequentialRNN:  # Language model with `arch` encoder and linear decoder
    "Create a language model from `arch` and its `config`."
    meta = core._model_meta[arch]
    config = ifnone(config, meta["config_lm"]).copy()
    for k in config.keys():
        if k.endswith("_p"):
            config[k] *= drop_mult
    tie_weights, output_p, out_bias = map(
        config.pop, ["tie_weights", "output_p", "out_bias"]
    )
    init = config.pop("init") if "init" in config else None
    encoder = arch(vocab_sz, **config)
    decoder = LinearDecoder(output_sz, config[meta["hid_name"]], output_p)
    model = SequentialRNN(encoder, decoder)
    return model if init is None else model.apply(init)
