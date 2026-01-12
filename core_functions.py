from numpy.fft import ifft2, fft2
from numpy import conj, pad, argmax, unravel_index, abs
from matplotlib import pyplot as plt

def fast_template_match(template, target, target_istransformed=False):
    """Fast template matching using Fourier Transforms

    :param template: Template image
    :param target: Target image
    """

    # Pad the template to match the target size
    template_H, template_W = template.shape[:2]
    target_H, target_W = target.shape[:2]   

    template_padded = pad(template, [(0, target_H - template_H), (0, target_W - template_W)], mode='constant', constant_values=0)

    # Compute the phase correlation
    if not target_istransformed:
        xcorr = fft2(target) * conj(fft2(template_padded))
    else:
        xcorr = target * conj(fft2(template_padded))
    phase_xcorr = ifft2(xcorr / abs(xcorr))
    
    # Find the location of maximum correlation
    idx_max = argmax(phase_xcorr)
    row, col = unravel_index(idx_max, target.shape)

    return (row, col)