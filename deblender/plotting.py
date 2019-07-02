import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gs


def make_plot(blended, true_x, true_y, gan_x, gan_y, savedir, batch):
    """
    Plots paneled figure of preblended, blended and deblended galaxies.
    """
    for i in range(blended.shape[0]):
        fig = plt.Figure()

        psnr_x = compare_psnr(im_test=gan_x[i], im_true=true_x[i])
        psnr_y = compare_psnr(im_test=gan_y[i], im_true=true_y[i])
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

        ax1.imshow(true_x[i])
        ax1.text(3., 10., r'Preblended 1', color='#FFFFFF')
        ax2.imshow(true_y[i])
        ax2.text(3., 10., r'Preblended 2', color='#FFFFFF')
        ax3.imshow(blended[i])
        ax3.text(1.3, 4.4, r'Blended', color='#FFFFFF')
        ax4.imshow(gan_x[i])
        ax4.text(3., 10., r'Deblended 1', color='#FFFFFF')
        ax4.text(3., 75., str(psnr_x)+' dB', color='#FFFFFF') #
        ax5.imshow(gan_y[i])
        ax5.text(3., 10., r'Deblended 2', color='#FFFFFF')
        ax5.text(3., 75., str(psnr_y)+' dB', color='#FFFFFF') #

        plt.tight_layout(pad=0)
        plt.subplots_adjust(wspace=0.06,hspace=-0.42)

        filename = os.path.join(savedir, 'test-{}-{}.png'.format(batch, i))
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
