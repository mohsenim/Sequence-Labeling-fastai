# Leveraging Language Models for Sequence Labeling Using fastai

Sequence labeling is a fundamental task in natural language processing (NLP). It aims to assign a label to each element in an input sequence. Examples of sequence labeling tasks include named entity recognition (NER) and noun phrase chunking. To annotate text using neural-based models, it's typical to split the input text into sentences or chunks to avoid memory issues that may occur when processing entire documents at once. However, this approach has the drawback of losing preceding information. The model implemented here memorizes the state of underlying language model and allows sequence labeling at the document-level.  

## Sequence Labeling Implementation Using fastai

Sequence labeling is not implemented in the fastai library. Inspired from the existing implementations of `language_model_learner` and `text_classification_learner` in fastai, I developed a sequence annotator.

The implemented model is built upon the `AWD-LSTM` language model to leverage its pre-trained capabilities. The language model analyzes words and its output sequence is then processed by a dense layer, which generates an output based on the representation vector of each input token.

I needed to define a new type for output classes, similar to `CategoryBlock`, capable of handling sequential data. This new type is called `SequentialCategoryBlock`. Additionally, a new data loader, `SequentialDataLoader`, was required to manage chunking, batching, and shuffling of data. `SequentialCategoryBlock` is called by `SequentialTextBlock`, another new class implemented to maintain consistency with `TextBlock`.

It is crucial to ensure that the number of tokens and labels match. Since the default tokenizer rules in fastai are not always reversible, tokenization must be performed before using the model.


## Dialogue Detection: A Case Study

For one of my studies, it was required to distinguish dialogues from non-dialogue texts. Dialogue detection, which can be defined as a sequence labeling task, is particularly important for analyzing literary and fictional texts, where direct speech between characters occur frequently. To accomplish dialogue detection, a model can be trained to analyze the text sequentially and annotate each word or token with tags indicating whether it belongs to a dialogue or not.

### Dataset

I used 100 fictional texts to prepare the data for training the model. These texts contain more 11.5M tokens. I assumed that quotation marks indicate the start and end of dialogues. Although the data may be noisy, the language model incorporated into the model is known to be tolerant to noise up to a high level.

For annotation, I used the IOB (Inside, Outside, Beginning) standards. 'B' signifies the beginning of dialogue text, 'I' denotes words continuing the dialogues, and 'O' represents words outside of dialogues (non-dialogue parts).

### Results

Training of the model like other language models was time-consuming, so for the first experiment I stuck to a relatively modest corpus size of only 100 texts. First, I trained the model for one epoch while keeping the underlying language model frozen and only training the upper linear layer. Then, I allowed the entire model to be trained for 10 epochs. The model achieved an accuracy of 99.4%. Not bad!
