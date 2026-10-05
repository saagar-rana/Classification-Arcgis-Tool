"""
LIDARLearn Models Package

Authors: LIDARLearn contributors
License: See repository LICENSE

Aggregate import module that registers every model class with the
MODELS registry. Optional CUDA-extension models are imported via
_safe_import so missing extensions degrade gracefully.
"""
import warnings
from .build import build_model_from_cfg

def _safe_import(module_path):
    """Import a model module, warn on failure (missing CUDA extension)."""
    try:
        __import__(module_path)
    except ImportError as e:
        warnings.warn(
            f"Could not import {module_path}: {e}. "
            f"Install the required extension to use this model.",
            stacklevel=2,
        )

# SSL backbones
import models.pointmae.pointmae
import models.pointbert.dvae
import models.pointbert.pointbert
import models.pointgpt.pointgpt
import models.pointgpt.pointgpt_seg
import models.act.act
import models.recon.recon
import models.pcpmae.pcpmae
import models.pcpmae.pcpmae_seg
import models.pointm2ae.pointm2ae
import models.pointm2ae.pointm2ae_pretrain
import models.pointm2ae.pointm2ae_seg

# PEFT strategies
import models.ppt.ppt
import models.ppt.ppt_seg
import models.idpt.idpt
import models.idpt.idpt_seg
import models.dapt.dapt
import models.dapt.dapt_seg
import models.dapt.pointgpt_dapt
import models.pointgst.pointgst
import models.pointgst.pointgst_seg
import models.pointgst.pointgpt_gst
import models.pointgpt.pointgpt_ppt
import models.pointgpt.pointgpt_idpt
import models.pointgpt.pointgpt_vpt_deep

# Supervised models (no CUDA extension dependencies)
import models.dgcnn.dgcnn
import models.pointnet.pointnet
import models.pointnet2.pointnet2
import models.pointnet2.pointnet2_seg
import models.pct.pct
import models.pct.pct_seg
import models.pointmlp.pointmlp
import models.pointmlp.pointmlp_seg
import models.pointconv.pointconv
import models.pointconv.pointconv_seg
import models.pointweb.pointweb
import models.pointweb.pointweb_seg
import models.curvenet.curvenet
import models.curvenet.curvenet_seg
import models.deepgcn.deepgcn
import models.deepgcn.deepgcn_seg
_safe_import("models.sonet.sonet")                         # index_max
import models.pointcnn.pointcnn
import models.pointkan.pointkan
import models.pointkan.pointkan_seg
import models.kandgcnn.kan_dgcnn
import models.gdan.gdan
import models.ppfnet.ppfnet
import models.pointscnet.network
import models.pointscnet.pointscnet_seg
import models.randlenet.randlenet
import models.randlenet.randlenet_seg
import models.rscnn.rscnn
import models.rscnn.rscnn_seg
import models.msdgcnn.msdgcnn
import models.msdgcnn.msdgcnn_seg
import models.msdgcnn2.msdgcnn2
import models.globaltransformer.globalltransformer
import models.globaltransformer.globaltransformer_seg
import models.pointtnt.pointtnt
import models.pointtnt.pointtnt_seg
_safe_import("models.p2p.p2p")                             # torch_scatter
_safe_import("models.p2p.p2p_seg")                         # torch_scatter
import models.rapsurf.rapsurf
import models.rapsurf.rapsurf_seg

# Models requiring optional CUDA extensions (graceful fallback)
_safe_import("models.dela.dela")                          # dela_cutils
_safe_import("models.dela.dela_seg")                      # dela_cutils
_safe_import("models.pvt.pvt")                            # ptv_modules
_safe_import("models.pvt.pvt_seg")                        # ptv_modules
_safe_import("models.pointtransformer.pointtransformer")      # pointops
_safe_import("models.pointtransformer.pointtransformer_seg")  # pointops
_safe_import("models.pointtransformerv2.pointtransformerv2")      # pointops
_safe_import("models.pointtransformerv2.pointtransformerv2_seg")  # pointops
_safe_import("models.pointtransformerv3.pointtransformerv3")      # spconv, torch_scatter
_safe_import("models.pointtransformerv3.pointtransformerv3_seg")  # spconv, torch_scatter

# Segmentation models
import models.seg.registered_seg
import models.seg.transformer_seg
