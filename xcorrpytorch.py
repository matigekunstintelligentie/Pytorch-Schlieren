import torch
import numpy as np
import cv2
from normxcorr2 import normxcorr2
from scipy.signal import fftconvolve, convolve2d
import time

def complex_multiplication(t1, t2):
    real1 = t1[:,:,0]
    real2 = t2[:,:,0]
    imag1 = t1[:,:,1]
    imag2 = t2[:,:,1]

    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

def opencv(original, template):
    return cv2.matchTemplate(original, template, cv2.TM_CCOEFF_NORMED)

def matlab(original_crop, current_crop):
    return normxcorr2(current_crop, original_crop,'valid')

def fft_convolve_pytorch(original, template, original_pre_fft=False, template_pre_fft=False, signal_sizes=None, template_size=None):
    if signal_sizes is None:
        signal_sizes = original.size()

    if template_size is None:
        template_size = template.size()

    if not original_pre_fft:
        original_fft = torch.rfft(original, 2)
    else:
        original_fft = original

    if not template_pre_fft:
        padded_template = torch.zeros_like(original)
        padded_template[:-(original.shape[0]-template.shape[0]),:-(original.shape[1]-template.shape[1])] = template
        template_fft = torch.rfft(padded_template, 2)
    else:
        template_fft = template

    return torch.irfft(complex_multiplication(original_fft, template_fft), 2, signal_sizes=signal_sizes)[template_size[0]-1:,template_size[1]-1:], original_fft, template_fft

def pytorch_fft(original, template):
    original = original - original.mean()
    template = template - template.mean()

    a1 = torch.ones_like(template)

    template_flip = torch.flip(template, [0,1])

    out, _, _ = fft_convolve_pytorch(original, template_flip)

    sq_term, _, _ = fft_convolve_pytorch(original**2, a1)
    normal_term, _, _ = fft_convolve_pytorch(original, a1)

    normal_term = (normal_term**2)/(template.size(0)*template.size(1))

    image = sq_term - normal_term

    image[image<0] = 0

    template = (template**2).sum()
    out = out/torch.sqrt(image*template)

    out[torch.logical_not(torch.isfinite(out))] = 0

    return out

def pytorch_conv2d(original, template):
    original = (original - original.mean()).unsqueeze(0).unsqueeze(0)
    template = (template - template.mean()).unsqueeze(0).unsqueeze(0)

    a1 = torch.ones_like(template)
    out = torch.nn.functional.conv2d(original, template, bias=None, padding=0, stride=(1, 1))

    image = torch.nn.functional.conv2d(original**2, a1, bias=None, padding=0, stride=(1, 1)) - ((torch.nn.functional.conv2d(original, a1, bias=None, padding=0, stride=(1, 1)))**2)/(np.prod(template.shape))

    image[image<0] = 0

    template = (template**2).sum()
    out = out/torch.sqrt(image*template)

    out[torch.logical_not(torch.isfinite(out))] = 0

    return out

def pytorch_fft_optimised(original, template):
    original = (original - original.mean(dim=(0,1)))

    template = (template - template.mean())

    a1 = torch.ones_like(template)

    template_flip = torch.flip(template, [0, 1])

    out, original_fft, _ = fft_convolve_pytorch(original, template_flip)

    sq_term, _, a1_fft = fft_convolve_pytorch(original**2, a1)

    normal_term, _, _ = fft_convolve_pytorch(original_fft, a1_fft, True, True, signal_sizes=original.size(), template_size=template.size())
    normal_term = (normal_term**2)/(template.size(0)*template.size(1))

    image = sq_term - normal_term

    image[image<0] = 0

    template = torch.mul(template, template).sum()

    out = out/torch.sqrt(image*template)

    out[torch.logical_not(torch.isfinite(out))] = 0

    return out

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
    with torch.cuda.amp.autocast():
        correlation_net = Correlation(template.size()).to("cuda")
        return correlation_net(original.to("cuda"), template.to("cuda"))

benchmark_value = 10000

original_crop = np.random.rand(benchmark_value, 64,64)*256
current_crop = np.random.rand(benchmark_value, 32,32)*256

original_crop_tensor = torch.from_numpy(original_crop).float()
current_crop_tensor = torch.from_numpy(current_crop).float()

print_result = False

print(torch.cuda.is_available())

t = time.time()
for i in range(benchmark_value):
    result = opencv(original_crop[i].astype(np.uint8), current_crop[i].astype(np.uint8))
    if i==0 and print_result:
        print(result[-1,-1])
print("opencv", time.time() - t)

t = time.time()
for i in range(benchmark_value):
    result = matlab(original_crop[i], current_crop[i])
    if i==0 and print_result:
        print(result[-1,-1])
print("matlab", time.time() - t)

t = time.time()
for i in range(benchmark_value):
    result = pytorch_conv2d(original_crop_tensor[i], current_crop_tensor[i])
    if i==0 and print_result:
        print(result[-1,-1,-1,-1])
print("pytorch conv2d", time.time() - t)

t = time.time()
for i in range(benchmark_value):
    result = pytorch_conv2d(original_crop_tensor[i].to("cuda"), current_crop_tensor[i].to("cuda"))
    if i==0 and print_result:
        print(result[-1,-1,-1,-1])
print("pytorch cuda conv2d", time.time() - t)

t = time.time()
for i in range(benchmark_value):
    result = pytorch_fft(original_crop_tensor[i], current_crop_tensor[i])
    if i==0 and print_result:
        print(result[-1,-1])
print("pytorch fft", time.time() - t)

t = time.time()
for i in range(benchmark_value):
    result = pytorch_fft(original_crop_tensor[i].to("cuda"), current_crop_tensor[i].to("cuda"))
    if i==0 and print_result:
        print(result[-1,-1])
print("pytorch cuda fft", time.time() - t)

t = time.time()
for i in range(benchmark_value):
    result = pytorch_fft_optimised(original_crop_tensor[i], current_crop_tensor[i])
    if i==0 and print_result:
        print(result[-1,-1])
print("pytorch fft optimised", time.time() - t)

t = time.time()
for i in range(benchmark_value):
    result = pytorch_fft_optimised(original_crop_tensor[i].to("cuda"), current_crop_tensor[i].to("cuda"))
    if i==0 and print_result:
        print(result[-1,-1])
print("pytorch fft cuda optimised", time.time() - t)

for j in [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]:
    print(j)
    t = time.time()
    for i in range(int(benchmark_value/j)):
        result = pytorch_fft_cuda(original_crop_tensor[i*j:(i+1)*j], current_crop_tensor[i*j:(i+1)*j])
        if print_result:
            print(result[0][-1,-1])
    print("pytorch fft cuda one by one", time.time() - t)
