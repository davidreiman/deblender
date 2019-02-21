import tfplot
import seaborn as sns
import matplotlib.pyplot as plt


def plot_spectrum(spec):
    """
    Plots 1-D data and returns matplotlib figure for use with tfplot.
    """
    plt.style.use('seaborn')
    fig, ax = tfplot.subplots(figsize=(4, 3))
    im = ax.plot(np.arange(1210, 1280, 0.25), spec)
    return fig
