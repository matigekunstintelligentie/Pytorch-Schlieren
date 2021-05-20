import cv2
import numpy as np
from normxcorr2 import normxcorr2
import matplotlib.pyplot as plt

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

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

            original_crop_eq = np.array_equal(original_crop,np.full(np.shape(original_crop),original_crop[0]))
            current_crop_eq = np.array_equal(current_crop,np.full(np.shape(current_crop),current_crop[0]))

            if original_crop_eq or current_crop_eq:
                rPeak = 0                                                   # Set row peak to zero
                cPeak = 0                                                   # Set column peak to zero
                dR    = 0                                                   # Set sub-pixel row delta to zero
                dC    = 0
            else:
                xcorr = normxcorr2(original_crop, current_crop)

                max_xcorr_index = np.unravel_index(xcorr.argmax(), xcorr.shape)
                row_peak = max_xcorr_index [1]
                col_peak = max_xcorr_index [0]
                xcorr = abs(xcorr)
                xcorr[xcorr == 0] = 0.0001
                col_xcorr, row_xcorr = np.shape(xcorr)

                if (row_peak == 0):
                    row_peak = row_peak + 1
                if (row_peak == row_xcorr-1):
                    row_peak = row_xcorr - 2
                if (col_peak == 0):
                    col_peak = col_peak + 1
                if (col_peak == col_xcorr-1):
                    col_peak = col_xcorr - 2

                numR = np.log(xcorr[col_peak][row_peak-1]) - np.log(xcorr[col_peak][row_peak+1])
                denR = 2*np.log(xcorr[col_peak][row_peak-1]) - 4*np.log(xcorr[col_peak][row_peak]) + 2*np.log(xcorr[col_peak][row_peak+1])
                dR = numR/denR

                numC = np.log(xcorr[col_peak-1][row_peak]) - np.log(xcorr[col_peak+1][row_peak])
                denC = 2*np.log(xcorr[col_peak-1][row_peak]) - 4*np.log(xcorr[col_peak][row_peak]) + 2*np.log(xcorr[col_peak+1][row_peak])
                dC = numC/denC

            row_peaks[i][j] = row_peak + dR
            col_peaks[i][j] = col_peak + dC

            row_offsets[i][j] = row_peaks[i][j] - window_size_half - search_size_half + 1
            col_offsets[i][j] = col_peaks[i][j] - window_size_half - search_size_half + 1           # Correct the column pixel shift for window and search sizes

    row_offsets[abs(row_offsets) > 1] = np.nan
    col_offsets[abs(col_offsets) > 1] = np.nan
    quivVel = np.sqrt(np.square(row_offsets) + np.square(col_offsets))


    plt.figure()
    plt.contourf(col_meshgrid,row_meshgrid,quivVel,100, cmap = "plasma", extend = 'both')
    plt.gca().invert_yaxis()                                            # Invert the Y-axis
    plt.gca().set_aspect('equal')
    plt.savefig("save/{}.png".format(idx))


if __name__=="__main__":
    original = preprocess(cv2.imread("jonathan/dots_30_t_ref.jpg"))
    current = cv2.imread("jonathan/dots_30_5.jpg")

    schlieren(original, current, 0)
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
