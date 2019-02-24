import tfplot
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import Gridspec as gs


def plot_spectrum(spec):
    """
    Plots 1-D data and returns matplotlib figure for use with tfplot.
    """
    plt.style.use('seaborn')
    fig, ax = tfplot.subplots(figsize=(4, 3))
    im = ax.plot(np.arange(1210, 1280, 0.25), spec)
    return fig


def make_plot(fig, blended, true_x, true_y, gan_x, gan_y, savedir, i, j):
    
    psnr_x = compare_psnr(im_test=gan_x, im_true=true_x)
    psnr_y = compare_psnr(im_test=gan_y, im_true=true_y)
    psnr_x = np.around(psnr_x, decimals=2)
    psnr_y = np.around(psnr_y, decimals=2)

    gs = GridSpec(2, 4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0:2, 1:3])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 3])

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.axis('off')

    ax1.imshow(true_x)
    ax1.text(3., 10., r'Preblended 1', color='#FFFFFF')
    ax2.imshow(true_y)
    ax2.text(3., 10., r'Preblended 2', color='#FFFFFF')
    ax3.imshow(blended)
    ax3.text(1.3, 4.4, r'Blended', color='#FFFFFF')
    ax4.imshow(gan_x)
    ax4.text(3., 10., r'Deblended 1', color='#FFFFFF')
    ax4.text(3., 75., str(psnr_x)+' dB', color='#FFFFFF') #
    ax5.imshow(gan_y)
    ax5.text(3., 10., r'Deblended 2', color='#FFFFFF')
    ax5.text(3., 75., str(psnr_y)+' dB', color='#FFFFFF') #

    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0.06,hspace=-0.42)

    filename = os.path.join(savedir, 'test-{}-{}.png'.format(i, j))
    plt.savefig(filename, dpi=300, bbox_inches='tight')
