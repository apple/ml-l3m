l3m.helpers.dist
================
.. module:: l3m.helpers.dist
.. currentmodule:: l3m.helpers.dist
.. autosummary::
    :toctree: _autosummary

    fsdp.init_fsdp2
    fsdp.build_grouping_plan
    fsdp.build_activation_checkpoint_plan
    fsdp.apply_ac_to_module
    fsdp.parallelize_model
    fsdp.apply_ac
    gather.all_gather_batch
    gather.all_gather_batch_with_grad
    tp.tp_parallelize
    cp.CPInputPlan
    cp.CPParameterPlan
    cp.CPPlan
    cp.build_cp_plan
    cp.create_context_parallel_ctx
    cp.get_train_context
    utils.setup_for_distributed
    utils.is_dist_avail_and_initialized
    utils.get_world_size
    utils.get_rank
    utils.get_local_rank
    utils.is_main_process
    utils.init_distributed_mode
    utils.DeviceMeshHandler
    utils.aggregate_tensor_across_devices
    utils.replicate_tensor_if_distributed
