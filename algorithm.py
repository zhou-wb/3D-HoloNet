import torch, math
from torch import nn
import imageio
import torch.optim as optim
from tqdm import tqdm
import utils

class DPAC(nn.Module):
    """Double-phase Amplitude Coding

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> dpac = DPAC(...)
    >>> _, final_phase = dpac(target_amp, target_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    target_amp (optional): phase at the target plane, with dimensions [batch, 1, height, width]
    final_phase: optimized phase-only representation at SLM plane, same dimensions

    """
    def __init__(self, prop_dist, wavelength, feature_size, prop_model='ASM', propagator=None,
                 device=torch.device('cuda')):
        """

        """
        super(DPAC, self).__init__()

        # propagation is from target to SLM plane (one step)
        self.prop_dist = -prop_dist
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.precomputed_H = None
        self.prop_model = prop_model
        self.prop = propagator
        self.dev = device

    def forward(self, target_amp, target_phase=None):
        if target_phase is None:
            target_phase = torch.zeros_like(target_amp)

        if self.precomputed_H is None and self.prop_model == 'ASM':
            self.precomputed_H = self.prop(torch.empty(*target_amp.shape, dtype=torch.complex64), self.feature_size,
                                           self.wavelength, self.prop_dist, return_H=True)
            self.precomputed_H = self.precomputed_H.to(self.dev).detach()
            self.precomputed_H.requires_grad = False

        final_phase = double_phase_amplitude_coding(target_phase, target_amp, self.prop_dist,
                                                    self.wavelength, self.feature_size,
                                                    prop_model=self.prop_model, propagator=self.prop,
                                                    precomputed_H=self.precomputed_H)
        return None, final_phase

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path

def double_phase_amplitude_coding(target_phase, target_amp, prop_dist, wavelength, feature_size,
                                  prop_model='ASM', propagator=None,
                                  dtype=torch.float32, precomputed_H=None):
    """
    Use a single propagation and converts amplitude and phase to double phase coding

    Input
    -----
    :param target_phase: The phase at the target image plane
    :param target_amp: A tensor, (B,C,H,W), the amplitude at the target image plane.
    :param prop_dist: propagation distance, in m.
    :param wavelength: wavelength, in m.
    :param feature_size: The SLM pixel pitch, in meters.
    :param prop_model: The light propagation model to use for prop from target plane to slm plane
    :param propagator: propagation_ASM
    :param dtype: torch datatype for computation at different precision.
    :param precomputed_H: pre-computed kernel - to make it faster over multiple iteration/images - calculate it once

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """
    real, imag = polar_to_rect(target_amp, target_phase)
    target_field = torch.complex(real, imag)

    slm_field = propagate_field(target_field, propagator, prop_dist, wavelength, feature_size,
                                      prop_model, dtype, precomputed_H)

    slm_phase = double_phase(slm_field, three_pi=False, mean_adjust=True)

    return slm_phase


def double_phase(field, three_pi=True, mean_adjust=True):
    """Converts a complex field to double phase coding

    field: A complex64 tensor with dims [..., height, width]
    three_pi, mean_adjust: see double_phase_amp_phase
    """
    return double_phase_amp_phase(field.abs(), field.angle(), three_pi, mean_adjust)

def double_phase_amp_phase(amplitudes, phases, three_pi=True, mean_adjust=True):
    """converts amplitude and phase to double phase coding

    amplitudes:  per-pixel amplitudes of the complex field
    phases:  per-pixel phases of the complex field
    three_pi:  if True, outputs values in a 3pi range, instead of 2pi
    mean_adjust:  if True, centers the phases in the range of interest
    """
    # normalize
    amplitudes = amplitudes / amplitudes.max()
    amplitudes = torch.clamp(amplitudes, -0.99999, 0.99999)

    # phases_a = phases - torch.acos(amplitudes)
    # phases_b = phases + torch.acos(amplitudes)
    
    acos_amp = torch.acos(amplitudes)
    phases_a = phases - acos_amp
    phases_b = phases + acos_amp

    phases_out = phases_a
    phases_out[..., ::2, 1::2] = phases_b[..., ::2, 1::2]
    phases_out[..., 1::2, ::2] = phases_b[..., 1::2, ::2]

    if three_pi:
        max_phase = 3 * math.pi
    else:
        max_phase = 2 * math.pi

    if mean_adjust:
        phases_out = phases_out - phases_out.mean()

    return (phases_out + max_phase / 2) % max_phase - max_phase / 2



# utils
def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


def replace_amplitude(field, amplitude):
    """takes a Complex tensor with real/imag channels, converts to
    amplitude/phase, replaces amplitude, then converts back to real/imag

    resolution of both Complex64 tensors should be (M, N, height, width)
    """
    # replace amplitude with target amplitude and convert back to real/imag
    real, imag = polar_to_rect(amplitude, field.angle())

    # concatenate
    return torch.complex(real, imag)


def propagate_field(input_field, propagator, prop_dist=0.2, wavelength=520e-9, feature_size=(6.4e-6, 6.4e-6),
                    prop_model='ASM', dtype=torch.float32, precomputed_H=None):
    """
    A wrapper for various propagation methods, including the parameterized model.
    Note that input_field is supposed to be in Cartesian coordinate, not polar!

    Input
    -----
    :param input_field: pytorch complex tensor shape of (1, C, H, W), the field before propagation, in X, Y coordinates
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength of the wave in m.
    :param feature_size: pixel pitch
    :param prop_model: propagation model ('ASM', 'MODEL', 'fresnel', ...)
    :param trained_model: function or model instance for propagation
    :param dtype: torch.float32 by default
    :param precomputed_H: Propagation Kernel in Fourier domain (could be calculated at the very first time and reuse)

    Output
    -----
    :return: output_field: pytorch complex tensor shape of (1, C, H, W), the field after propagation, in X, Y coordinates
    """

    if prop_model == 'ASM':
        output_field = propagator(u_in=input_field, z=prop_dist, feature_size=feature_size, wavelength=wavelength,
                                  dtype=dtype, precomped_H=precomputed_H)
    else:
        raise ValueError('Unexpected prop_model value')

    return output_field



def gradient_descent(init_phase, target_amp, target_mask=None, forward_prop=None, num_iters=1000, roi_res=None,
                     loss_fn=nn.MSELoss(), lr=0.01, out_path_idx='./results',
                     citl=False, camera_prop=None, writer=None, admm_opt=None,
                     *args, **kwargs):
    """
    Gradient-descent based method for phase optimization.

    :param init_phase:
    :param target_amp:
    :param target_mask:
    :param forward_prop:
    :param num_iters:
    :param roi_res:
    :param loss_fn:
    :param lr:
    :param out_path_idx:
    :param citl:
    :param camera_prop:
    :param writer:
    :param args:
    :param kwargs:
    :return:
    """

    assert forward_prop is not None
    dev = init_phase.device
    num_iters_admm_inner = 1 if admm_opt is None else admm_opt['num_iters_inner']

    slm_phase = init_phase.requires_grad_(True)  # phase at the slm plane
    optvars = [{'params': slm_phase}]
    optimizer = optim.Adam(optvars, lr=lr)

    loss_vals = []
    loss_vals_quantized = []
    best_loss = 10.
    best_iter = 0

    if target_mask is not None:
        target_amp = target_amp * target_mask
        nonzeros = target_mask > 0
    if roi_res is not None:
        target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False)
        if target_mask is not None:
            target_mask = utils.crop_image(target_mask, roi_res, stacked_complex=False)
            nonzeros = target_mask > 0

    if admm_opt is not None:
        u = torch.zeros(1, 1, *roi_res).to(dev)
        z = torch.zeros(1, 1, *roi_res).to(dev)

    for t in range(num_iters):
        for t_inner in range(num_iters_admm_inner):
            optimizer.zero_grad()

            recon_field = forward_prop(slm_phase)
            recon_field = utils.crop_image(recon_field, roi_res, stacked_complex=False)
            recon_amp = recon_field.abs()

            if citl:  # surrogate gradients for CITL
                captured_amp = camera_prop(slm_phase)
                captured_amp = utils.crop_image(captured_amp, roi_res,
                                                stacked_complex=False)
                recon_amp = recon_amp + captured_amp - recon_amp.detach()

            if target_mask is not None:
                final_amp = torch.zeros_like(recon_amp)
                final_amp[nonzeros] += (recon_amp[nonzeros] * target_mask[nonzeros])
            else:
                final_amp = recon_amp

            with torch.no_grad():
                s = (final_amp * target_amp).mean() / \
                    (final_amp ** 2).mean()  # scale minimizing MSE btw recon and

            loss_val = loss_fn(s * final_amp, target_amp)

            # second loss term if ADMM
            if admm_opt is not None:
                # augmented lagrangian
                recon_phase = recon_field.angle()
                loss_prior = loss_fn(utils.laplacian(recon_phase) * target_mask, (z - u) * target_mask)
                loss_val = loss_val + admm_opt['rho'] * loss_prior

            loss_val.backward()
            optimizer.step()

        ## ADMM steps
        if admm_opt is not None:
            with torch.no_grad():
                reg_norm = utils.laplacian(recon_phase).detach() * target_mask
                Ax = admm_opt['alpha'] * reg_norm + (1 - admm_opt['alpha']) * z  # over-relaxation
                z = utils.soft_thresholding(u + Ax, admm_opt['gamma'] / (rho + 1e-10))
                u = u + Ax - z

                # varying penalty (rho)
                if admm_opt['varying-penalty']:
                    if t == 0:
                        z_prev = z

                    r_k = ((reg_norm - z).detach() ** 2).mean()  # primal residual
                    s_k = ((rho * utils.laplacian(z_prev - z).detach()) ** 2).mean()  # dual residual

                    if r_k > admm_opt['mu'] * s_k:
                        rho = admm_opt['tau_incr'] * rho
                        u /= admm_opt['tau_incr']
                    elif s_k > admm_opt['mu'] * r_k:
                        rho /= admm_opt['tau_decr']
                        u *= admm_opt['tau_decr']
                    z_prev = z

        with torch.no_grad():
            if loss_val < best_loss:
                best_phase = slm_phase
                best_loss = loss_val
                best_amp = s * recon_amp
                best_iter = t + 1
    print(f' -- optimization is done, best loss: {best_loss}')

    return {'loss_vals': loss_vals,
            'loss_vals_q': loss_vals_quantized,
            'best_iter': best_iter,
            'best_loss': best_loss,
            'recon_amp': best_amp,
            'final_phase': best_phase}