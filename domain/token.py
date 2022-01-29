class Token(object):
    
    def __init__(self, begin, end, text):
        self.begin = begin
        self.end = end
        self.text = text
        self.original_text = text
        self.subtokens = []
        self.predicted_label = None

    def __repr__(self):
        return f"Token({self.text}, {self.original_text}, {self.begin}, {self.end})"
        
    def __str__(self):
        return self.__repr__()
