from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import tensorflow as tf

from .document_processor import DocumentProcessor


class T5Processor:

    def __init__(self, model="t5-base"):
        self.model = TFT5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        # T5 uses a max_length of 512 so we cut the article to 450 tokens to allow t5 'normalization'
        self.doc_processor = DocumentProcessor()
        self.input_ids = None
        self.embeddings = {'word': None, 'doc': None, 'paragraphs': None}
        self.outputs = None

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

    def paragraph_embeddings(self):
        if self.embeddings['paragraphs'] is None:
            self.embeddings['paragraphs'] = tf.reduce_mean(self.embeddings['word'], axis=1)
        return self.embeddings['paragraphs']

    def doc_embeddings(self):
        if self.embeddings['paragraphs'] is None:
            self.paragraph_embeddings()
        if self.embeddings['doc'] is None:
            self.embeddings['doc'] = tf.reduce_mean(self.embeddings['paragraphs'], axis=0)
        return self.embeddings['doc']

    # T5 text2text summarization (requires .fit(..., task="summarize:")
    def summarize(self, max_length=125, min_length=None):
        self.outputs = self.model.generate(self.input_ids, max_length, min_length=min_length,
                                           length_penalty=2.0, num_beams=4, early_stopping=True)
        return self

    def print(self):
        for ids in self.outputs:
            print(self.tokenizer.decode(ids))
        return self

