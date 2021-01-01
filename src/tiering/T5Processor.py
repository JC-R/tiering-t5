from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import tensorflow as tf
import numpy as np

from .document_processor import DocumentProcessor


# splice a tensor
def tf_splice(tensor, _start=0, _end=None):
    start = tf.constant(0) if _start is None else tf.constant(_start)
    end = tf.constant(len(tensor)) if _end is None else tf.constant(_end)
    return tensor[start:end]


class T5Processor:

    def __init__(self, model="t5-base", batch_size=32):
        self.model = TFT5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        # T5 uses a max_length of 512 so we cut the article to 450 tokens to allow t5 'normalization'
        self.doc_processor = DocumentProcessor()
        self.input_ids = None
        self.embeddings = {'word': None, 'doc': None, 'paragraphs': None}
        self.outputs = None
        self.batch_size = batch_size

    def split(self, document, max_words=450, truncate=False, task=""):
        return self.doc_processor.split(document, max_words=max_words, truncate=truncate, prefix=task)

    #  generate embeddings
    def fit(self, document, split=True, truncate=True, task=""):
        input_doc = self.split(document, task=task) if split else document
        tokens = self.tokenizer(input_doc, return_tensors="tf", truncation=truncate, padding=True)
        self.input_ids = tokens['input_ids']
        encodings = self.model.encoder(self.input_ids, attention_mask=tokens['attention_mask'])

        #  use mean pooling layer to produce a document embedding (this is what sentence_transformers does)
        self.embeddings['word'] = encodings[0]
        return self

    # compute paragraph embeddings given a range or all
    def paragraph_embeddings(self, _start=0, _end=None):
        if self.embeddings['paragraphs'] is None:
            tensor = tf_splice(self.embeddings['word'], _start=_start, _end=_end)
            self.embeddings['paragraphs'] = tf.reduce_mean(tensor, axis=1)
        return self.embeddings['paragraphs']

    # compute doc embeddings [subgroups, if given a range]
    def doc_embeddings(self, _start=0, _end=None):
        if self.embeddings['paragraphs'] is None:
            self.paragraph_embeddings(_start=_start, _end=_end)
        if self.embeddings['doc'] is None:
            tensor = tf_splice(self.embeddings['paragraphs'], _start=_start, _end=_end)
            self.embeddings['doc'] = tf.reduce_mean(tensor, axis=0)
        return self.embeddings['doc']

    #
    def get_embeddings(self, batch, groups=None):
        s = len(batch)
        start = 0
        end = self.batch_size
        embeddings = None
        while start < s:
            self.fit(batch[start:end], split=False)
            self.paragraph_embeddings()
            if embeddings is None:
                embeddings = self.embeddings['paragraphs'].numpy()
            else:
                embeddings = np.append(embeddings, self.embeddings['paragraphs'].numpy(), axis=0)
            start += self.batch_size
            end += self.batch_size
        return embeddings

    # T5 text2text summarization (requires .fit(..., task="summarize:")
    def summarize(self, max_length=125, min_length=None):
        self.outputs = self.model.generate(self.input_ids, max_length, min_length=min_length,
                                           length_penalty=2.0, num_beams=4, early_stopping=True)
        return self

    def print(self):
        for ids in self.outputs:
            print(self.tokenizer.decode(ids))
        return self
