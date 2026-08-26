"""Audio-separation workflow step."""

from muvisual_workflow.separation.bs_roformer_sw import (
    AUDIO_EXTENSIONS,
    apply_noise_gate,
    create_separator,
    main,
    prepare_local_model,
    separate_with_loaded_model,
)

__all__ = [
    "AUDIO_EXTENSIONS",
    "apply_noise_gate",
    "create_separator",
    "main",
    "prepare_local_model",
    "separate_with_loaded_model",
]
