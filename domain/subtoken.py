class SubToken(object):
    
    def __init__(self, token, text):
        self.token = token
        self.text = text
        self.predicted_label = None

    def __repr__(self):
        return f"SubToken({self.text}, token={self.token})"
        
    def __str__(self):
        return self.__repr__()
