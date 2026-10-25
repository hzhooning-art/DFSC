"""Explicit optional CPU backend; does not register or replace dfsc.solve."""
from .high_precision_cpu import RepairRefusal,RepairResult,evaluate as evaluate_scalar
from .selective_spectral import ActionRefusal,ActionResult,SelectiveSpectral

__version__='0.1.0.dev0'
__all__=['RepairRefusal','RepairResult','evaluate_scalar','ActionRefusal','ActionResult',
         'SelectiveSpectral','backend_info']


def backend_info():
    return {'distribution':'dfsc-cpu-repair-candidate','version':__version__,
            'namespace':'dfsc_cpu_repair','device':'cpu','dtype':'float64',
            'scalar_alpha_range':[.45,1.95],'spectral_alpha_range':[.45,1.],
            'maximum_modal_radius':16.,'first_order_only':True,
            'operator_or_time_gradients':False,'automatic_dfsc_registration':False,
            'rigorous_floating_point_certificate':False,'public_release':False}
