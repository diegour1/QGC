def kernel_density_classification(X_train_param, Y_train_oh_param, X_test_param, gamma_param = 4.):

  ## Initialize values
  Y_train_oh_param = np.array(Y_train_oh_param)
  num_classes_temp = Y_train_oh_param.shape[1]
  Y_pred_temp = np.zeros((len(X_test_param), num_classes_temp))
  sigma_temp = 1./(np.sqrt(2.*gamma_param))

  ## Training
  kde_temp = []
  for j in range(num_classes_temp):
    kde_temp.append(KernelDensity(kernel='gaussian', bandwidth=sigma_temp).fit(X_train_param[Y_train_oh_param[:, j].astype(bool)]))
  ## Prediction
  for j in range(num_classes_temp):
    Y_pred_temp[:, j] =  (Y_train_oh_param[:, j].sum()/len(Y_train_oh_param))*np.exp(kde_temp[j].score_samples(X_test_param))
  
  return Y_pred_temp