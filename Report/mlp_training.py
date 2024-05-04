import numpy as np
import pandas as pd
import re
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from itertools import product
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Helper functions
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

def get_permutations_with_replacement(elements, n):
    """
    Generate all permutations of `elements` of size `n` with replacement.
    
    Parameters:
    - elements: List of integers.
    - n: Size of each permutation.
    
    Returns:
    - List of tuples, each tuple is a permutation of `n` elements.
    """
    return list(product(elements, repeat=n))

# So we can see the data without it being truncated
pd.set_option('display.max_colwidth', None)

data = pd.read_csv("clean_dataset.csv")

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

t = np.array(data["Label"])

# Split data into test, valid, and test set
X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=2/10, stratify=t)
X_train, X_valid, t_train, t_valid= train_test_split(X_tv, t_tv, test_size=2/10, stratify=t_tv)

# # Tuning hyperparameters
best_train = 0
best_valid = 0
best_test = 0
best_n = 0
coef = None
intercept = None

for i in range(1, 101):
  print(f"Training model with {i} neurons...")
  clf = MLPClassifier(hidden_layer_sizes=(i,), activation="tanh", solver="sgd", max_iter=5000)
  clf.fit(X_train, t_train)

  train_acc = clf.score(X_train, t_train)
  valid_acc = clf.score(X_valid, t_valid)
  test_acc = clf.score(X_test, t_test)

  if valid_acc > best_valid:
    best_test = test_acc
    best_valid = valid_acc
    best_train = train_acc
    best_n = i
    coef = clf.coefs_.copy()
    intercept = clf.intercepts_.copy()

print(f"Training complete\n")

print(f"Best number of neurons: {best_n}")
print(f"Best train accuracy: {best_train}")
print(f"Best valid accuracy: {best_valid}")
print(f"Best test accuracy: {best_test}")

np.save('weights.npy', np.array(coef, dtype=object), allow_pickle=True)
np.save('biases.npy', np.array(intercept, dtype=object), allow_pickle=True)

np.save("X_train.npy", X_train, allow_pickle=True)
np.save("t_train.npy", t_train, allow_pickle=True)
np.save("X_valid.npy", X_valid, allow_pickle=True)
np.save("t_valid.npy", t_valid, allow_pickle=True)
np.save("X_test.npy", X_test, allow_pickle=True)
np.save("t_test.npy", t_test, allow_pickle=True)
