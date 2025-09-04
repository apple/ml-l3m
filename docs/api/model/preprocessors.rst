l3m.model.preprocessors
=======================
.. module:: l3m.model.preprocessors
.. currentmodule:: l3m.model.preprocessors
.. autosummary::
    :toctree: _autosummary

    text.TextPreprocessor
    vision.ViTPreprocessor
    vision.ViTPreprocessorWithRegisters
    vision.OpenAICLIPViTProcessor
    vision.OpenAICLIPViTProcessorWithRegisters
    vision.PatchEmbed
    vision.PatchEmbedLinear
    vision.TokenSubSampler
    vision.TextPreprocessor
    multimodal.VisionAndTextPreprocessor
    transforms.PadToSquare
    transforms.expand_to_square
    mask_builders.MaybeCausalAttentionMaskBuilder
    mask_builders.PrefixAttentionMaskBuilder
    mask_builders.MultimodalDecoderMaskBuilder
    mask_builders.FlexAttnPrefixAttentionMaskBuilder
    mask_builders.PaddingAttentionMaskBuilder

l3m.model.preprocessors.pos_embed
---------------------------------
.. module:: l3m.model.preprocessors.pos_embed
.. currentmodule:: l3m.model.preprocessors.pos_embed
.. autosummary::
    :toctree: _autosummary

    rope.RoPE
    sinusoidal.get_2d_sincos_pos_embed
    sinusoidal.get_3d_sincos_pos_embed
