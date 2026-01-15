from numpy.fft import ifft2, fft2
from numpy import conj, pad, argmax, unravel_index, abs
from matplotlib import pyplot as plt
import pyfftw

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

def faster_template_match(template, target, target_istransformed=False):
    """Faster template matching using pyfftw

    :param template: Template image
    :param target: Target image
    """

    # Pad the template to match the target size
    template_H, template_W = template.shape[:2]
    target_H, target_W = target.shape[:2]   

    template_padded = pad(template, [(0, target_H - template_H), (0, target_W - template_W)], mode='constant', constant_values=0)

    template_padded = pyfftw.byte_align(template_padded, n=16)

    # Compute the phase correlation
    if not target_istransformed:
        target = pyfftw.byte_align(target, n=16)
        xcorr = pyfftw.interfaces.numpy_fft.fft2(target) * conj(pyfftw.interfaces.numpy_fft.fft2(template_padded))
    else:
        xcorr = target * conj(pyfftw.interfaces.numpy_fft.fft2(template_padded))
    phase_xcorr =  pyfftw.interfaces.numpy_fft.ifft2(xcorr / abs(xcorr))
    
    # Find the location of maximum correlation
    idx_max = argmax(phase_xcorr)
    row, col = unravel_index(idx_max, target.shape)

    return (row, col)