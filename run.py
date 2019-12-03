import logging
import utils as utils

TREES = 2
SET_NAME = 'titanic'  # Check list of usable sets in the folder "db"
LOG_LEVEL = 1  # Use 0 to print all messages, 1 to 'light print' and 2 to mute
K_FOLDS = 10
DEPTH = None


def main():
    logging.getLogger().setLevel(LOG_LEVEL*60)
    x, y = utils.read_data(SET_NAME)

    ''' Compare RF vs WF training '''
    # utils.kfold_regression_mse(x, y, t_method='RF', n_folds=K_FOLDS, n_trees=TREES, n_state=2000, m_depth=DEPTH)
    utils.kfold_regression_mse(x, y, t_method='WF', num_wavelets=100000, n_state=2000, n_folds=K_FOLDS,
                               m_depth=DEPTH, n_trees=TREES)

    print('lolz')

    ''' Check optimal m-term '''
    # mse_m, best_term, best_norm = utils.find_m_term(x, y, trees=TREES, folds=K_FOLDS, budget=20,
    #                                                 method='hop', depth=DEPTH)
    # utils.plot_vec(mse_m[:, 2], mse_m[:, 0], 'MSE vs #wavelets', '#wavelets', 'MSE')

    ''' Sort features by importance '''
    # print(utils.sort_features_by_importance(x, y, t_method='RF', n_trees=TREES, m_depth=DEPTH))
    # print(utils.sort_features_by_importance(x, y, t_method='WF', n_trees=TREES, n_threshold=0.1, m_depth=DEPTH,
    #                                         norms_normalization='samples'))

    ''' Check mse adding features one by one (after sorting by importance) '''
    # mse_rf = utils.kfold_error_one_by_one_feature(x, y, method='RF', trees=TREES, depth=DEPTH)
    # mse_wf = utils.kfold_error_one_by_one_feature(x, y, method='WF', trees=TREES, depth=DEPTH,
    #                                               wavelets=best_term, threshold=best_norm, nnormalization='samples')
    # utils.plot_2vec(mse_rf, mse_wf, 'MSE vs Additive features using random and wavelet forest',
    #                 '#first most important features', 'MSE')


if '__main__' == __name__:
    main()
