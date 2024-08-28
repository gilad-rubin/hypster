
#This is a mock class for demonstration purposes
class LLMModel:
    def __init__(self, chunking, model, config):
        self.chunking = chunking
        self.model = model
        self.config = config
    
    def __eq__(self, other):
        return (self.chunking == other.chunking and
                self.model == other.model and
                self.config == other.config)
