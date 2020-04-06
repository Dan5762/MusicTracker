class Error(Exception):
    
    def __init__(self, msg):
        self.msg = msg


class ProcessingError(Error):
    pass
