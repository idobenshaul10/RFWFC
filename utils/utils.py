import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from tree_models.random_forest import WaveletsForestRegressor
from tree_models.dyadic_decision_tree_regressor import DyadicDecisionTreeRegressor
import torchvision

def normalize_data(x_raw):
    x = (x_raw - np.min(x_raw, 0))/(np.max(x_raw, 0) - np.min(x_raw, 0))
    x = np.nan_to_num(x)
    return x

def train_model(x, y, method='RF', mode='regression', trees=5, depth=9, features='auto',
                state=2000, threshold=1000, train_vi=False, nnormalization='volume', cube_length=1.):

    if method == 'RF':
        model = RandomForestRegressor(n_estimators=trees, max_depth=depth, \
                                      max_features=features, random_state=state)
    elif method == 'WF':
        model = WaveletsForestRegressor(regressor='random_forest', \
            mode=mode, trees=trees, depth=depth, train_vi=train_vi, features=features, \
            seed=state, vi_threshold=threshold, norms_normalization=nnormalization, cube_length=cube_length)

    elif method == "dyadic":
        model = DyadicDecisionTreeRegressor(depth=depth, seed=state, \
            norms_normalization=nnormalization, cube_length=cube_length)

    else:
        raise Exception('Method incorrect - should be either RF or WF')    
    model.fit(x, y)
    return model


def predict_model(x, model, method='RF', m=10):
    if method == 'RF':
        return model.predict(x)
    elif method == 'WF':
        return model.predict(x, m)
    else:
        raise Exception('Method incorrect - should be either RF or WF')

def run_alpha_smoothness(X, y, t_method='RF', num_wavelets=1000, n_trees=1, m_depth=9,
                         n_features='auto', n_state=2000, normalize=True, \
                         norm_normalization='volume', cube_length=1., error_TH=0.1, \
                         text='', output_folder='', epsilon_1=None, epsilon_2=None):
    
    if normalize:
        X = normalize_data(X)

    norm_m_term = 0    
    model = train_model(X, y, method=t_method, trees=n_trees, \
        depth=m_depth, features=n_features, state=n_state, \
        nnormalization=norm_normalization, cube_length=cube_length)

    if t_method == 'WF':
        if num_wavelets < 1:
            num_wavelets = int(np.round(num_wavelets*len(model.norms)))
            norm_m_term = -np.sort(-model.norms)[num_wavelets-1]

    # alpha, n_wavelets, errors = model.evaluate_smoothness(m=num_wavelets, error_TH=error_TH)    
    alpha = model.evaluate_angle_smoothness(text=text, \
        output_folder=output_folder, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
    n_wavelets, errors = 0., 0.

    logging.log(60, 'ALPHA SMOOTHNESS over X: ' + str(alpha))
    return alpha, -1, num_wavelets, norm_m_term, model


def save_wavelet_norms(X, y, t_method='RF', num_wavelets=1000, n_trees=1, m_depth=9,
                         n_features='auto', n_state=2000, normalize=True, \
                         norm_normalization='volume', cube_length=1.):
    if normalize:
        X = normalize_data(X)

    norm_m_term = 0
    model = train_model(X, y, method=t_method, trees=n_trees, \
        depth=m_depth, features=n_features, state=n_state, \
        nnormalization=norm_normalization, cube_length=cube_length)  

    model.save_wavelet_norms() 

def calculate_besov_semi_norm(X, y, t_method='RF', num_wavelets=1000, n_trees=1, m_depth=9,
                         n_features='auto', n_state=2000, normalize=True, \
                         norm_normalization='volume', cube_length=1.):
    if normalize:
        X = normalize_data(X)

    norm_m_term = 0
    model = train_model(X, y, method=t_method, trees=n_trees, \
        depth=m_depth, features=n_features, state=n_state, \
        nnormalization=norm_normalization, cube_length=cube_length)  

    model.calculate_besov_semi_norm()  


def kfold_alpha_smoothness(x, y, t_method='RF', num_wavelets=10, n_folds=10, n_trees=5, m_depth=9,
                         n_features='auto', n_state=2000, normalize=True, norm_normalization='volume'):
    
    if normalize:
        x = normalize_data(x)
    
    kf = KFold(n_splits=n_folds)
    alphas = []
    np.random.seed(seed=n_state)
    shuffle_data = np.arange(len(x))
    np.random.shuffle(shuffle_data)

    norm_m_term = 0
    with tqdm(total=n_folds) as pbar:
        for idx, (train, test) in enumerate(kf.split(x)):            
            x_train = x[shuffle_data[train]]
            y_train = y[shuffle_data[train]]            
            model = train_model(x_train, y_train, method=t_method, trees=n_trees,
                        depth=m_depth, features=n_features, state=n_state, \
                        nnormalization=norm_normalization)
            if t_method == 'WF':
                if num_wavelets < 1:
                    num_wavelets = int(np.round(num_wavelets*len(model.norms)))
                    norm_m_term = -np.sort(-model.norms)[num_wavelets-1]

            alpha, n_wavelets, errors = model.evaluate_smoothness(m=num_wavelets)            
            alphas.append(alpha)
            logging.log(20, f'Fold {idx} alpha: {str(alphas[-1])}')            
            pbar.update(1)

    logging.log(60, 'MEAN ALPHA SMOOTHNESS over all folds: ' + str(np.mean(alphas)) +
                ' STD: ' + str(np.std(alphas)))
    return np.mean(alphas), np.std(alphas), num_wavelets, norm_m_term


def kfold_regression_mse(x, y, t_method='RF', num_wavelets=10, n_folds=5, n_trees=5, m_depth=9,
                         n_features='auto', n_state=2000, normalize=True, norm_normalization='volume'):
    # Input - Labeled data and number of folds
    # Output - Mean and standard deviation of mean squared errors over all folds

    # Normalize the data if needed
    if normalize:
        x = normalize_data(x)

    # Use scikit-learn's KFold, will automatically split the data into training and testing in each fold
    kf = KFold(n_splits=n_folds)
    mse = []

    # Shuffle the data indexes to get k-random folds
    np.random.seed(seed=n_state)
    shuffle_data = np.arange(len(x))
    np.random.shuffle(shuffle_data)

    norm_m_term = 0

    for train, test in kf.split(x):
        # Create the training and testing arrays for each fold
        x_train = x[shuffle_data[train]]
        y_train = y[shuffle_data[train]]
        x_test = x[shuffle_data[test]]
        y_test = y[shuffle_data[test]]
        model = train_model(x_train, y_train, method=t_method, trees=n_trees,
                            depth=m_depth, features=n_features, state=n_state, nnormalization=norm_normalization)
        if t_method == 'WF':
            if num_wavelets < 1:
                num_wavelets = int(np.round(num_wavelets*len(model.norms)))
                norm_m_term = -np.sort(-model.norms)[num_wavelets-1]
        y_pred = predict_model(x_test, model, method=t_method, m=num_wavelets)        
        mse.append(metrics.mean_squared_error(y_test, y_pred))
        
    return np.mean(mse), np.std(mse), num_wavelets, norm_m_term


def find_m_term(x, y, budget=100, folds=10, trees=5, depth=9, features='auto', state=2000, method='fixed',
                nnormalization='volume'):
    mse_m = np.zeros((budget, 4))
    for k in range(0, budget):
        logging.log(60, ' Using ' + method + ' in ' + str(k) + ' iteration.')
        if method == 'hop':
            wavelets = (k+1)/budget - np.finfo(float).eps
        else:
            wavelets = k+1
        mse_m[k, 0], mse_m[k, 1], mse_m[k, 2], mse_m[k, 3] = kfold_regression_mse(x, y, t_method='WF',
                                                          num_wavelets=wavelets, n_folds=folds,
                                                          n_trees=trees, m_depth=depth,
                                                          n_features=features, n_state=state,
                                                          norm_normalization=nnormalization)
    return mse_m, int(mse_m[np.argmin(mse_m[:, 0]), 2]), mse_m[np.argmin(mse_m[:, 0]), 3]


def sort_features_by_importance(x, y, t_method='RF', n_trees=5, m_depth=9, normalize=True,
                                n_features='auto', n_state=2000, n_threshold=1000, norms_normalization='volume'):
    if normalize:
        x = normalize_data(x)
    model = train_model(x, y, method=t_method, trees=n_trees, depth=m_depth, train_vi=True,
                        features=n_features, state=n_state, threshold=n_threshold, nnormalization=norms_normalization)
    return np.argsort(-model.feature_importances_)


def kfold_error_one_by_one_feature(x, y, method='RF', trees=5, depth=9, features='auto', state=2000,
                                   wavelets=1000, threshold=0, nnormalization='volume'):
    logging.log(60, '   Adding features one-by-one sorted by VI using ' + method)
    sorted_vec = sort_features_by_importance(x, y, t_method=method, n_trees=trees, m_depth=depth, n_features=features,
                                             n_state=state, n_threshold=threshold, norms_normalization=nnormalization)
    new_x = x[:, sorted_vec]
    mse = []
    for k in range(0, new_x.shape[1]):
        mse.append(kfold_regression_mse(new_x[:, 0:(k+1)], y, t_method=method, n_trees=trees, m_depth=depth,
                                        n_features=features, n_state=state, num_wavelets=wavelets,
                                        norm_normalization=nnormalization)[0])
    return mse

def imshow(img):
    # img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def visualize_augmentation(images):    
    imshow(torchvision.utils.make_grid(images))

if __name__ == '__main__':    
    from sklearn.datasets import make_moons, make_circles, make_classification
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

    import pdb; pdb.set_trace()