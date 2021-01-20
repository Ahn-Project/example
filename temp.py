def Direct_Strategy():
    RMSE_byStep=[]
    hist_byStep = []
    runtime_byStep = []
    actual_tests = []
    predicted_tests = []

    # Direct strategy
    for i in range(n_outputs):
            dataset = load(1)
            dataset_X, dataset_Y = \
                seq2dataset(dataset, X_col_name, y_col_name, n_inputs, n_outputs)
            # X 만들기
            direct_X = dataset_X.copy()
            std_direct_X = std(direct_X)
            # Y 만들기
            temp_y = dataset_Y.copy()
            temp_y = temp_y[:,i]
            direct_y = np.reshape(np.array(temp_y), (dataset_Y.shape[0], 1))


            # 모델 학습
            train_X, train_y, val_X, val_y, test_X, test_y = \
                split_data(dataset, std_direct_X, direct_y)
            model_TVOC, hist_TVOC, RMSE_TVOC, actual_test_X, \
            actual_test_TVOC, predicted_test_TVOC, runtime \
                = Modeling(train_X, train_y, val_X, val_y, test_X, test_y)
            actual_tests.append(actual_test_TVOC)
            predicted_tests.append(predicted_test_TVOC)
            RMSE_byStep.append(RMSE_TVOC)
            runtime_byStep.append(runtime[0])


    RMSE = round(np.mean(RMSE_byStep), 2)

    p = np.array(predicted_tests)
    a = np.array(actual_tests)

    for i in range(len(p)):
        if i==0:
            actual_test = a[0]
            predicted_test = p[0]
        else:
            actual_test = np.concatenate((actual_test,a[i]), axis=1)
            predicted_test = np.concatenate((predicted_test,p[i]), axis=1)
    return actual_test, predicted_test, RMSE_byStep, runtime_byStep
