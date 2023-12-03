import pickle
import hazm
import re

class ClassificationModel():
  def __init__(self):
    filename= "model.pickle"
    self.model=pickle.load(open(filename,"rb"))

    self.normalizer = hazm.Normalizer()
    self.lemmatizer = hazm.Lemmatizer()
  def normalize_custom(self, line: str):
      line = re.sub(r'[.{}[\]؛:«»؟!٬٫٪×،*)(ـ+<>\'",`=+\-?!@#$%^&*()_\/\\\\]', '', line.strip())
      line = re.sub(r'\s+', ' ', line.strip())
      line = self.normalizer.normalize(line)
      words = hazm.word_tokenize(line)
      words = [self.lemmatizer.lemmatize(word) for word in words]
      line = ' '.join(words)
      return line

  def classify_text(self, test_dataframe):
  # computation of the model’s output
    test_queries=test_dataframe.reset_index()["text"]
    for i in range(len(test_queries)):
      test_queries[i]=self.normalize_custom(test_queries[i])

    return self.model.predict(test_queries)