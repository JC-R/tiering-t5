import re


class DocumentProcessor():

    def __init__(self):
        # allow some growth
        self.bad_content = "bad utf-8 encoding"
        self.remove_chars = re.compile(r'[\n\|/\x00-\x09\x0c\x0e-0x1f\x80-\xff]')
        self.words = re.compile(r'\w+')
        self.spaces = re.compile(r'  +')
        self.doc_groups = []

    # split the input into maxsize words partitions
    def split(self, body, truncate=False, max_words=512, prefix=""):
        if prefix is None:
            prefix = ""
        tokens = self.words.findall(body)
        size = len(tokens)
        if size <= max_words:
            self.doc_groups.append(1)
            return [prefix + body]
        if truncate:
            self.doc_groups.append(1)
            return [prefix + " ".join(tokens[0:min(max_words, len(tokens)) - 1])]
        else:
            partitions = []
            start = 0
            end = max_words - 1
            while start <= size - 1:
                partitions.append(prefix + " ".join(tokens[start:end]))
                start += max_words
                end += max_words
            self.doc_groups.append(len(partitions))
        return partitions

    def clear(self):
        self.doc_groups.clear()
