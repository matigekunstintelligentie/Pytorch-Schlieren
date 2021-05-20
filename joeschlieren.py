import cv2
import numpy as np
from normxcorr2 import normxcorr2
import matplotlib.pyplot as plt
import time
import torch

# TODO: batching, i.e. wait to fill up buffer
# Different window, search sizes
# Unet implementation
# Torch script

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def gaussian_estimator(xcorr, col_peak, row_peak):
    numR = np.log(xcorr[col_peak][row_peak-1]) - np.log(xcorr[col_peak][row_peak+1])
    denR = 2*np.log(xcorr[col_peak][row_peak-1]) - 4*np.log(xcorr[col_peak][row_peak]) + 2*np.log(xcorr[col_peak][row_peak+1])
    dR = numR/denR

    numC = np.log(xcorr[col_peak-1][row_peak]) - np.log(xcorr[col_peak+1][row_peak])

    denC = 2*np.log(xcorr[col_peak-1][row_peak]) - 4*np.log(xcorr[col_peak][row_peak]) + 2*np.log(xcorr[col_peak+1][row_peak])

    dC = numC/denC

    return row_peak + dR, col_peak + dC

def parabolic_estimator(xcorr, col_peak, row_peak):
    numR = xcorr[col_peak][row_peak-1] - xcorr[col_peak][row_peak+1]
    denR = 2*xcorr[col_peak][row_peak-1] - 4*xcorr[col_peak][row_peak] + 2*xcorr[col_peak][row_peak+1]
    dR = numR/denR

    numC = xcorr[col_peak-1][row_peak] - xcorr[col_peak+1][row_peak]
    denC = 2*xcorr[col_peak-1][row_peak] - 4*xcorr[col_peak][row_peak] + 2*xcorr[col_peak+1][row_peak]
    dC = numC/denC
    return row_peak + dR, col_peak + dC

def peak_centroid_estimator(xcorr, col_peak, row_peak):
    numR = (row_peak-1)*xcorr[col_peak][row_peak-1] + (row_peak)*xcorr[col_peak][row_peak] + (row_peak+1)*xcorr[col_peak][row_peak+1]
    denR = xcorr[col_peak][row_peak-1] + xcorr[col_peak][row_peak] + xcorr[col_peak][row_peak+1]
    dR = numR/denR

    numC = (col_peak-1)*xcorr[col_peak-1][row_peak] + (col_peak)*xcorr[col_peak][row_peak] + (col_peak+1)*xcorr[col_peak+1][row_peak]
    denC = xcorr[col_peak-1][row_peak] + xcorr[col_peak][row_peak] + xcorr[col_peak+1][row_peak]
    dC = numC/denC

    return dR, dC

def three_point_estimator(xcorr, col_peak, row_peak, estimator='Gaussian'):
    # Particle Image Velocimetry: A Practical Guide table 5.1
    if estimator=="Gaussian":
        return gaussian_estimator(xcorr, col_peak, row_peak)
    elif estimator=="Parabolic":
        return parabolic_estimator(xcorr, col_peak, row_peak)
    else:
        return peak_centroid_estimator(xcorr, col_peak, row_peak)




class Correlation(torch.nn.Module):
    def __init__(self, template_size):
        super(Correlation, self).__init__()
        self.a1 = torch.ones(template_size)

    def forward(self, original, template):
        normalised_original = original - original.mean(dim=(1,2)).view(-1,1,1)

        normalised_template = template - template.mean(dim=(1,2)).view(-1,1,1)

        normalised_template_flip = torch.flip(normalised_template, [1, 2])

        out, original_fft, _ = self.fft_convolve_pytorch(normalised_original, normalised_template_flip)

        sq_term, _, a1_fft = self.fft_convolve_pytorch(normalised_original**2, self.a1)

        normal_term, _, _ = self.fft_convolve_pytorch(original_fft, a1_fft, True, True, signal_sizes=normalised_original.size(), template_size=normalised_template.size())

        normal_term = (normal_term**2)/(template.size(1)*template.size(2))

        image = sq_term - normal_term

        image[image < 0] = 0

        template = torch.einsum("ijk,ijk->i", normalised_template, normalised_template)

        out = out/torch.sqrt(image*template.view(-1,1,1))

        out[torch.logical_not(torch.isfinite(out))] = 0

        return out

    def fft_convolve_pytorch(self, original, template, original_pre_fft=False, template_pre_fft=False, signal_sizes=None, template_size=None):
        if signal_sizes is None:
            signal_sizes = original.size()

        signal_sizes = signal_sizes[1:]

        if template_size is None:
            template_size = template.size()

        template_size = template_size[1:]

        if not original_pre_fft:
            original_fft = torch.rfft(original, 2)
        else:
            original_fft = original

        if not template_pre_fft:
            padded_template = torch.zeros_like(original)
            padded_template[:, :-(original.shape[1]-template.shape[1]),:-(original.shape[2]-template.shape[2])] = template
            template_fft = torch.rfft(padded_template, 2)
        else:
            template_fft = template

        return torch.irfft(self.complex_multiplication(original_fft, template_fft), 2, signal_sizes=signal_sizes)[:,template_size[0]-1:,template_size[1]-1:], original_fft, template_fft

    def complex_multiplication(self, t1, t2):
        real1 = t1[:,:,:,0]
        real2 = t2[:,:,:,0]
        imag1 = t1[:,:,:,1]
        imag2 = t2[:,:,:,1]

        return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

def pytorch_fft_cuda(original, template):
    correlation_net = Correlation(template.size()).to("cuda")
    return correlation_net(original.to("cuda"), template.to("cuda"))


def gaussian_estimato_pytorch(xcorr, indices):
    numR = np.log(xcorr[col_peak][row_peak-1]) - np.log(xcorr[col_peak][row_peak+1])
    denR = 2*np.log(xcorr[col_peak][row_peak-1]) - 4*np.log(xcorr[col_peak][row_peak]) + 2*np.log(xcorr[col_peak][row_peak+1])
    dR = numR/denR

    numC = np.log(xcorr[col_peak-1][row_peak]) - np.log(xcorr[col_peak+1][row_peak])
    denC = 2*np.log(xcorr[col_peak-1][row_peak]) - 4*np.log(xcorr[col_peak][row_peak]) + 2*np.log(xcorr[col_peak+1][row_peak])
    dC = numC/denC

    return row_peak + dR, col_peak + dC

def schlieren_pytorch(original_frame, current_frame, idx=0):
    preprocessed_original_frame = torch.from_numpy(original_frame)
    preprocessed_current_frame = torch.from_numpy(preprocess(current_frame))

    n_rows = preprocessed_original_frame.size(0)
    n_cols = preprocessed_original_frame.size(1)

    window_size = 32
    window_size_half = int(np.floor(window_size/2))
    search_size = 64
    search_size_half = int(np.floor(search_size/2))

    window_row_indices = np.arange(window_size,(n_rows-window_size_half),window_size)
    window_col_indices = np.arange(window_size,(n_cols-window_size_half),window_size)

    row_meshgrid, col_meshgrid = np.meshgrid(window_row_indices, window_col_indices)

    row_meshgrid = np.transpose(row_meshgrid)
    col_meshgrid = np.transpose(col_meshgrid)

    original_crops = []
    current_crops = []

    for i in range(len(window_row_indices)):
        for j in range(len(window_col_indices)):
            row_center = window_row_indices[i]
            col_center = window_col_indices[j]


            current_crop = preprocessed_current_frame[row_center - search_size_half: row_center + search_size_half, col_center - search_size_half: col_center + search_size_half]
            original_crop = preprocessed_original_frame[row_center - window_size_half:row_center + window_size_half, col_center - window_size_half: col_center + window_size_half]

            original_crops.append(original_crop.unsqueeze(0).float())
            current_crops.append(current_crop.unsqueeze(0).float())

    print(torch.cat(current_crops).size(), torch.cat(original_crops).size())

    xcorr = pytorch_fft_cuda(torch.cat(current_crops), torch.cat(original_crops))

    batch_size = xcorr.size(0)
    xcorr_size = xcorr.size(1)

    i_xcorr = xcorr.view(batch_size , -1).argmax(1)
    peaks = torch.cat(((i_xcorr  // xcorr_size).view(-1, 1), (i_xcorr  % xcorr_size).view(-1, 1)), dim=1)

    xcorr = torch.abs(xcorr)
    xcorr[xcorr == 0] = np.finfo(float).eps

    row_xcorr, col_xcorr = list(xcorr.size())[1:]

    peaks[peaks==0] = peaks[peaks==0] + 1

    row_peaks = peaks[:,1]
    col_peaks = peaks[:,0]



    #zip these somehow
    numR = torch.log(xcorr[torch.arange(0,batch_size).long(),col_peaks.view(-1), row_peaks.view(-1)-1]) - torch.log(xcorr[torch.arange(0,batch_size).long(),col_peaks.view(-1), row_peaks.view(-1)+1])
    denR = 2*torch.log(xcorr[torch.arange(0,batch_size).long(),col_peaks.view(-1), row_peaks.view(-1)-1]) - 4*torch.log(xcorr[torch.arange(0,batch_size).long(),col_peaks.view(-1), row_peaks.view(-1)]) + 2*torch.log(xcorr[torch.arange(0,batch_size).long(),col_peaks.view(-1), row_peaks.view(-1)+1])
    dR = numR/denR

    dR = dR.view(-1,len(window_row_indices),len(window_col_indices))

    numC = torch.log(xcorr[torch.arange(0,batch_size).long(),col_peaks.view(-1)-1, row_peaks.view(-1)]) - torch.log(xcorr[torch.arange(0,batch_size).long(),col_peaks.view(-1)+1, row_peaks.view(-1)])
    denC = 2*torch.log(xcorr[torch.arange(0,batch_size).long(),col_peaks.view(-1)-1, row_peaks.view(-1)]) - 4*torch.log(xcorr[torch.arange(0,batch_size).long(),col_peaks.view(-1), row_peaks.view(-1)]) + 2*torch.log(xcorr[torch.arange(0,batch_size).long(),col_peaks.view(-1)+1, row_peaks.view(-1)])
    dC = numC/denC

    dC = dC.view(-1,len(window_row_indices),len(window_col_indices))

    row_peaks = row_peaks.view(len(window_row_indices),len(window_col_indices)) + dR.squeeze(0) - window_size_half
    col_peaks = col_peaks.view(len(window_row_indices),len(window_col_indices)) + dC.squeeze(0) - window_size_half

    row_peaks[row_peaks.abs() > 1.5] = row_peaks.mean()
    col_peaks[col_peaks.abs() > 1.5] = col_peaks.mean()


    quivVel = torch.sqrt(torch.pow(row_peaks, 2) + torch.pow(col_peaks, 2)).cpu().detach().numpy()

    mappings = ['viridis', 'plasma', 'inferno']
    # ,'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    #         'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    #         'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    #         'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'twilight', 'twilight_shifted', 'hsv', 'Pastel1', 'Pastel2', 'Paired', 'Accent',
    #                     'Dark2', 'Set1', 'Set2', 'Set3',
    #                     'tab10', 'tab20', 'tab20b', 'tab20c', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    #         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    #         'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

    for mapping in mappings:

        plt.figure()

        fig = plt.contourf(col_meshgrid,row_meshgrid,quivVel,1000, cmap = mapping, extend = 'both')
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.gca().set_aspect('equal')
        plt.savefig("save/{0}{1}.png".format(mapping,idx), bbox_inches='tight')
        plt.close()

def schlieren(original_frame, current_frame, idx=0):
    preprocessed_original_frame = original_frame
    preprocessed_current_frame = preprocess(current_frame)

    n_rows = preprocessed_original_frame.shape[0]
    n_cols = preprocessed_original_frame.shape[1]

    window_size = 32
    window_size_half = int(np.floor(window_size/2))
    search_size = 64
    search_size_half = int(np.floor(search_size/2))

    window_row_indices = np.arange(window_size,(n_rows-window_size_half),window_size)
    window_col_indices = np.arange(window_size,(n_cols-window_size_half),window_size)

    row_meshgrid, col_meshgrid = np.meshgrid(window_row_indices, window_col_indices)

    row_meshgrid = np.transpose(row_meshgrid)
    col_meshgrid = np.transpose(col_meshgrid)



    col_peaks = np.zeros((len(window_row_indices),len(window_col_indices)))
    row_peaks = np.zeros((len(window_row_indices),len(window_col_indices)))
    col_offsets = np.zeros((len(window_row_indices),len(window_col_indices)))
    row_offsets = np.zeros((len(window_row_indices),len(window_col_indices)))

    for i in range(len(window_row_indices)):
        for j in range(len(window_col_indices)):
            row_center = window_row_indices[i]
            col_center = window_col_indices[j]

            original_crop = preprocessed_original_frame[row_center - window_size_half:row_center + window_size_half, col_center - window_size_half: col_center + window_size_half]
            current_crop = preprocessed_current_frame[row_center - search_size_half: row_center + search_size_half, col_center - search_size_half: col_center + search_size_half]

            xcorr = cv2.matchTemplate(original_crop, current_crop, cv2.TM_CCOEFF_NORMED)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(xcorr)

            row_peak = maxLoc[1]
            col_peak = maxLoc[0]
            xcorr = abs(xcorr)
            xcorr[xcorr == 0] = np.finfo(float).eps
            col_xcorr, row_xcorr = np.shape(xcorr)

            if (row_peak == 0):
                row_peak = row_peak + 1
            if (row_peak == row_xcorr-1):
                row_peak = row_xcorr - 2
            if (col_peak == 0):
                col_peak = col_peak + 1
            if (col_peak == col_xcorr-1):
                col_peak = col_xcorr - 2

            row_peaks[i][j], col_peaks[i][j] = three_point_estimator(xcorr, col_peak, row_peak, 'Gaussian')

    row_offsets = row_peaks - window_size_half
    col_offsets = col_peaks - window_size_half  # Correct the column pixel shift for window and search sizes

    row_offsets[abs(row_offsets) > 1.5] = row_offsets.mean()
    col_offsets[abs(col_offsets) > 1.5] = col_offsets.mean()


    quivVel = np.sqrt(np.square(row_offsets) + np.square(col_offsets))


    mappings = ['viridis', 'plasma', 'inferno']
    # ,'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    #         'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    #         'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    #         'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'twilight', 'twilight_shifted', 'hsv', 'Pastel1', 'Pastel2', 'Paired', 'Accent',
    #                     'Dark2', 'Set1', 'Set2', 'Set3',
    #                     'tab10', 'tab20', 'tab20b', 'tab20c', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    #         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    #         'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

    for mapping in mappings:

        plt.figure()

        fig = plt.contourf(col_meshgrid,row_meshgrid,quivVel,1000, cmap = mapping, extend = 'both')
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.gca().set_aspect('equal')
        plt.savefig("save/og{0}{1}.png".format(mapping,idx), bbox_inches='tight')
        plt.close()

def compare(original_frame, current_frame, idx=0):
    out_schlieren_pytorch = schlieren_pytorch(original_frame, current_frame, idx=0)
    out_schlieren = schlieren(original_frame, current_frame, idx=0)

    print(np.sum(out_schlieren_pytorch-out_schlieren))

if __name__=="__main__":
    ims = ["jonathan/0.jpg"] #, "jonathan/1.jpg", "jonathan/2.jpg", "jonathan/3.jpg"]


    t = time.time()
    for idx, im in enumerate(ims):
        original = preprocess(cv2.imread("jonathan/reference.jpg"))
        current = cv2.imread(im)
        ret_schlier = schlieren(original, current, idx)
    print("schlieren", time.time()-t)

    t = time.time()
    for idx, im in enumerate(ims):
        original = preprocess(cv2.imread("jonathan/reference.jpg"))
        current = cv2.imread(im)
        ret_schlier_pt = schlieren_pytorch(original, current, idx)
    print("pytorch schlieren", time.time()-t)
    print(ret_schlier - ret_schlier_pt.cpu().numpy())

    quit()
    video = "MAH04544.MP4"

    cap = cv2.VideoCapture(video)

    i = 0
    while(True):
        print(i)
        ret, current = cap.read()

        if not ret:
            break

        schlieren(original, current, i)
        i+=1

    cap.release()
    cv2.destroyAllWindows()
