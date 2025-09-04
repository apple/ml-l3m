l3m.helpers
===========
.. module:: l3m.helpers
.. currentmodule:: l3m.helpers

.. toctree::
    :maxdepth: 1

    dist
    vision
    moe

.. autosummary::
    :toctree: _autosummary

    checkpoint.prepare_and_load_checkpoints
    checkpoint.load_checkpoint
    checkpoint.filter_checkpoint
    checkpoint.save_on_master
    checkpoint.rename_checkpoint
    checkpoint.dist_execute_only_main
    checkpoint.DistributedCheckpointUtils
    utils.move_to_device
    utils.get_chunk_from_elem
    utils.chunk_batch
    utils.RandomBinaryChoice
    utils.get_module
    utils.set_module
    utils.has_module
    utils.init_parameters
