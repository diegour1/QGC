import qmc.tf.layers as qmc_layers
import qmc.tf.models as qmc_models
import tensorflow as tf
import math



N_FFS = 32


def dmkde_classical_qrff(**kwargs):
    type_ffs = "qrff"
    if type_ffs == "rff":
        fm_x = qmc_layers.QFeatureMapRFF(DIM_X, dim=N_FFS, gamma=GAMMA/2, random_state=RANDOM_STATE_QRFF)
        qmd = qmc_models.QMDensity(fm_x, N_FFS)
    elif type_ffs == "qrff":
        fm_x = qmc_layers.QFeatureMapComplexRFF(DIM_X, dim=N_FFS, gamma=GAMMA/2, random_state= RANDOM_STATE_QRFF)
        qmd = qmc_models.ComplexQMDensity(fm_x, N_FFS)
    qmd.compile()
    qmd.fit(X_train, epochs=1)
    #qmd.fit(X_train[600:601], epochs=1) # uncomment for single point prediction

    predictions_classical = tf.cast(tf.math.pow((GAMMA/(tf.constant(math.pi))), DIM_X/2)*qmd.predict(X_plot), tf.float32).numpy()
    predictions_classical_train = tf.cast(tf.math.pow((GAMMA/(tf.constant(math.pi))), DIM_X/2)*qmd.predict(X_train), tf.float32).numpy()
    predictions_classical_test = tf.cast(tf.math.pow((GAMMA/(tf.constant(math.pi))), DIM_X/2)*qmd.predict(X_test), tf.float32).numpy()

    return predictions_classical_train, predictions_classical_test, predictions_classical