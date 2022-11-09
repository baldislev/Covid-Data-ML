from sklearn.base import BaseEstimator,ClassifierMixin

class kNN(BaseEstimator, ClassifierMixin):
  def __init__(self, n_neighbors:int = 5):
    self.n_neighbors = n_neighbors
  
  def fit(self, X, y):
    self.X_set = np.copy(X)
    self.y_label = np.copy(y)
    return self
    
  def predict(self, X):
    train_X, y = self.X_set, self.y_label
    predictions = None
    # Compute the predicted labels (+1 or -1)
    dist_mat = scipy.spatial.distance.cdist(X, train_X, metric='euclidean')
    top_k = [np.argsort(x, kind = 'quick_sort')[:self.n_neighbors] for x in dist_mat]
    top_k_labels = np.array([y[i] for i in top_k])
    # Most common label:
    predictions = [Counter(labels).most_common(1)[0][0] for labels in top_k_labels]
    return np.array(predictions)