import numpy as np
#import scipy.ndimage as ndimage
def my_imfilter(image, imfilter):

    '''
    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    '''
    ###################################################################################
    # TODO:                                                                           #
    # This function is intended to behave like the scipy.ndimage.filters.correlate    #
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         #
    # of the filter matrix.)                                                          #
    # Your function should work for color images. Simply filter each color            #
    # channel independently.                                                          #
    # Your function should work for filters of any width and height                   #
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       #
    # restriction makes it unambigious which pixel in the filter is the center        #
    # pixel.                                                                          #
    # Boundary handling can be tricky. The filter can't be centered on pixels         #
    # at the image boundary without parts of the filter being out of bounds. You      #
    # should simply recreate the default behavior of scipy.signal.convolve2d --       #
    # pad the input image with zeros, and return a filtered image which matches the   #
    # input resolution. A better approach is to mirror the image content over the     #
    # boundaries for padding.                                                         #
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can # 
    # see the desired behavior.                                                       #
    # When you write your actual solution, you can't use the convolution functions    #
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   #
    # Simply loop over all the pixels and do the actual computation.                  #
    # It might be slow.                                                               #
    ###################################################################################
    ###################################################################################
    # NOTE:                                                                           #
    # Some useful functions                                                           #
    #     numpy.pad or numpy.lib.pad                                                  #
    # #################################################################################
    #filter dimension
    m, n = imfilter.shape

    #image dimension
    x, y, z = image.shape
    x_o, y_o, z_o = image.shape

    #padding number and zero padding
    pad_w = (x - (x - m + 1))/2
    pad_h = (y - (y - n + 1))/2
    pad_image = np.zeros((x+2*pad_w,y+2*pad_h,z))
    for pad_ch in range(z):
        pad_image[:,:,pad_ch] = np.pad(image[:,:,pad_ch], ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values = ((0,0),(0,0)))

    #output image size
    x, y, z = pad_image.shape
    output = np.zeros((x_o,y_o,z_o))#new

    #convolution
    x_run = x - m + 1
    y_run = y - n + 1
    for i in range(x_run):
        for j in range(y_run):
            for k in range(z):
                output[i,j,k] = np.sum(pad_image[i:i+m, j:j+n, k]*imfilter)
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can 
    # see the desired behavior.

    ###################################################################################
    #                                 END OF YOUR CODE                                #
    ###################################################################################
    return output
