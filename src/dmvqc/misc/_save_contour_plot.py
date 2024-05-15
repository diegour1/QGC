import matplotlib.pyplot as plt

params = {
   'axes.labelsize': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [7.0, 6.0]
   }

def save_contour_plot(x, y, predictions):
    plt.rcParams.update(params)

    plt.contourf(x, y, predictions.reshape([120,120]))
    plt.colorbar()
    plt.savefig("spiralsdmkde_mixed_QRFF.pdf")