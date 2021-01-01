from sentence_transformers import SentenceTransformer


class Sentence2Vec():

    # if device=None, let system pick (GPU first)
    def __init__(self, modelpath, device=None, batch_size=32, num_workers=1):
        self.model = SentenceTransformer(modelpath, device=device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch = 0

    def get_embeddings(self, batch, groups=None):
        return self.model.encode(batch, self.batch_size, num_workers=self.num_workers)
