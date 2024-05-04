# imports
import numpy as np
import pandas as pd
import re

weights = np.load("weights.npy", allow_pickle=True)
biases = np.load("biases.npy", allow_pickle=True)

# Helper functions used for feature encoding
def to_numeric(s):
  """Converts string `s` to a float.

  Invalid strings and NaN values will be converted to 0.
  """
  if isinstance(s, str):
    s = s.replace(",", '')
    s = pd.to_numeric(s, errors="coerce")
  return float(s)

def get_number_list(s):
  """Get a list of integers contained in string `s`
  """
  return [int(n) for n in re.findall(r"(\d+)", str(s))]

def get_number(s):
  """Get the first number contained in string `s`.

  If `s` does not contain any numbers, return 0.
  """
  n_list = get_number_list(s)
  return n_list[0] if len(n_list) >= 1 else 0

def cat_in_s(s, cat):
  """Return if a category is present in string `s` as an binary integer.
  """
  return int(cat in s) if not pd.isna(s) else 0

def get_cat_ranking(s, cat):
  ranks = s.split(",")

  for r in ranks:
    if cat in r:
      return get_number(r)

  raise Exception("Error get_cat_ranking: invalid category")

def clean_quote(s):
  if isinstance(s, float):
    return ''

  lower = s.lower()
  parse_white_space = re.sub(r'\s+', ' ', lower)
  final = re.sub(r'[^a-z ]', '', parse_white_space)

  return final

def is_word_in_s(s, word):
  return int(word in clean_quote(s))

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.mean()

# Prediction functions
def predict(x):
  labels = ["Dubai", "New York City", "Paris", "Rio de Janeiro"]
  
  for i in range(len(weights)):
    w = np.transpose(weights[i])
    b = biases[i]
    x = np.matmul(w, x) - b
    if i != len(weights) - 1:
      x = np.tanh(x)

  y = np.argmax(softmax(x))
  return labels[y]

def predict_all(filename):
  data = pd.read_csv(filename)
  X = np.stack([
    data['Q1'].apply(to_numeric).fillna(0),
    data['Q2'].apply(to_numeric).fillna(0),
    data['Q3'].apply(to_numeric).fillna(0),
    data['Q4'].apply(to_numeric).fillna(0),
    data['Q5'].apply(lambda s: cat_in_s(s, "Partner")),
    data['Q5'].apply(lambda s: cat_in_s(s, "Friends")),
    data['Q5'].apply(lambda s: cat_in_s(s, "Siblings")),
    data['Q5'].apply(lambda s: cat_in_s(s, "Co-worker")),
    data['Q6'].apply(lambda s: get_cat_ranking(s, "Skyscrapers")),
    data['Q6'].apply(lambda s: get_cat_ranking(s, "Sport")),
    data['Q6'].apply(lambda s: get_cat_ranking(s, "Art and Music")),
    data['Q6'].apply(lambda s: get_cat_ranking(s, "Carnival")),
    data['Q6'].apply(lambda s: get_cat_ranking(s, "Cuisine")),
    data['Q6'].apply(lambda s: get_cat_ranking(s, "Economic")),
    data['Q7'].apply(to_numeric).fillna(0),
    data['Q10'].apply(lambda s: is_word_in_s(s, "dubai")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "rich")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "city")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "rio")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "love")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "dreams")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "paris")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "york")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "football")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "tower")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "concrete")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "money")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "baguette")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "brazil")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "carnival")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "jungle")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "eiffel")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "habibi")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "oil")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "big")),
    data['Q10'].apply(lambda s: is_word_in_s(s, "apple")),
  ], axis=1)

  return [predict(x) for x in X]
