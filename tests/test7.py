from hypster import Options, lazy


class Class:
    def __init__(self, ngram_range, analyzer, lowercase):
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.lowercase = lowercase

ngram_range = Options({"small" : (1, 3), "large" : (2, 8)}, default="small")
analyzer = Options(["char", "word"], default="char")

vectorizer = lazy(Class)(ngram_range=ngram_range, analyzer=analyzer, lowercase=True)

top_k = Options({"small" : 4, "large" : 6}, default="small")

final_vars = ["vectorizer", "top_k"] #TODO: find this automatically?