class Document(object):
    
    def __init__(self, doc_id, text, tokens, spans=None):
        self.id = doc_id
        self.text = text
        self.subtokens = None
        self.tokens = tokens
        self.tags = None
        self.spans = spans

    def __repr__(self):
        return f"Document({self.text})"
        
    def __str__(self):
        return self.__repr__()
