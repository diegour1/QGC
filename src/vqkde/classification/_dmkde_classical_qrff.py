def classical_dmkdc_qrff(X_train_param, Y_train_oh_param, X_test_param, n_qrffs_param, rs_param, gamma_param = 4.):

  ## Initialize values
  dim_x_temp = X_train_param.shape[1]
  Y_train_oh_param = np.array(Y_train_oh_param)
  num_classes_temp = Y_train_oh_param.shape[1]
  Y_pred_temp = np.zeros((len(X_test_param), num_classes_temp))
  sigma_temp = 1./(np.sqrt(2.*gamma_param))

  ## Training
  dmkdc_temp = []
  for j in range(num_classes_temp):
    fm_x = qmc_layers.QFeatureMapComplexRFF(dim_x_temp, dim=n_qrffs_param, gamma=gamma_param/2, random_state= rs_param)
    dmkdc_temp.append(qmc_models.ComplexQMDensity(fm_x, n_qrffs_param))
  
    ## Prediction
  for j in range(num_classes_temp):
    dmkdc_temp[j].compile()
    dmkdc_temp[j].fit(X_train_param[Y_train_oh_param[:, j].astype(bool)], epochs=1)
    Y_pred_temp[:, j] = ((gamma_param/np.pi)**(dim_x_temp/2))*(dmkdc_temp[j].predict(X_test_param))

  return ((Y_train_oh_param.sum(axis=0)/len(Y_train_oh_param)))*Y_pred_temp