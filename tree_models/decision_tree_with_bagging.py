import numpy as np
from sklearn import tree, linear_model, ensemble
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
import operator
import code
from tqdm import tqdm
from functools import reduce

class DecisionTreeWithBaggingRegressor:
  def __init__(self, criterion='mse', bagging=0.8, depth=9, trees=5, seed=None):
    self.criterion = criterion
    self.bagging = bagging
    self.trees = trees
    self.seed = seed
    self.random_state = np.random.RandomState(seed=self.seed)
    self.estimators_ = []
    if depth == -1:
      depth = None
    for i in range(trees):
      self.estimators_.append(tree.DecisionTreeRegressor(        
        max_depth=depth,
        max_features='auto',
        random_state = self.random_state,
      ))      

  def fit(self, X_all, y_all):
    indices_all = np.arange(len(X_all))

    random_state = np.random.RandomState(seed=self.seed)
    for i in tqdm(range(self.trees)):
      X, X_test, y, y_test, indices, indices_test = train_test_split(X_all, y_all, indices_all, test_size=1-self.bagging, random_state=random_state)
      X, y = shuffle(X, y, random_state=self.seed)
      self.estimators_[i].fit(X, y)
      cur_depth = self.estimators_[i].tree_.max_depth
      if cur_depth> 20:
        print(f"depth of tree is {cur_depth}")
        
    return self

  def decision_path(self, X):
    paths = []
    for i in range(self.trees):
      current_paths = self.estimators_[i].decision_path(X)      
      paths = np.append(paths, current_paths)

    paths_csr_dim = reduce(lambda s, x: s+x.shape[1], paths, 0)
    paths_csr = lil_matrix((np.shape(X)[0], paths_csr_dim),dtype=np.float32)
    current_i = 0
    for current_paths in paths:
      paths_csr[:, current_i:current_i + current_paths.shape[1]] = current_paths
      current_i += current_paths.shape[1]

    return paths_csr.tocsr(), []
  



