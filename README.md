# Large Multi-Modal Models (``l3m``)

<p align="center">
  <img src="docs/_static/logo.png" width="250"/>
</p>

``l3m`` is a flexible library for training any type of large model, regardless of modality. Instead of more traditional
approaches, we opt for a config-heavy approach, where each model training corresponds to a single .yaml file.
This makes reproducibility a first-class citizen, as it is easy to share configs, as well as making the code as
abstracted as possible from the general user. It also offers significant flexibility, making it easy to combine
different "building blocks" in a lego-like fashion.

It is mainly been used to train AIMv1 and AIMv2, but it can be used for training any type of model.
- **AIMv2**: [`Multimodal Autoregressive Pre-training of Large Vision Encoders`](https://arxiv.org/abs/2411.14402) [**CVPR 2025 - Highlight**]<br>
  Enrico Fini*, Mustafa Shukor*, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju,
  Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi,
  Yinfei Yang, Joshua M. Susskind, and Alaaeldin El-Nouby*.

- **AIMv1**: [`Scalable Pre-training of Large Autoregressive Image Models`](https://arxiv.org/abs/2401.08541) [**ICML 2024**]<br>
  Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar,
  Joshua M Susskind, Armand Joulin.

*: Equal technical contribution.

## Installation
Installing ``l3m`` requires the latest versions of torch (2.7) and some particular dependencies depending on the usage.
The full installation can be done as:
```commandline
conda create --name l3m python=3.10
conda activate l3m
pip install -e .
```

## Quickstart
``l3m`` relies heavily on config files (using `hydra` and `.yaml` files), we suggest that before starting to use ``l3m``
the user should get familiar with how to create configs.  We understand that this might be confusing at first, so we
provide a set of base configs in `configs` that will help with getting acquainted with ``l3m``.

In ``l3m`` configs are a 1-to-1 mapping of experiments. It fully represents a training run and nothing
is hidden in the code - only boilerplate code.

We provide [configs](configs) for AIMv1, AIMv2 (and AIMv2 with MoEs), CLIP, and a default LLM.

## Usage
Launch a training run as:
```bash
torchrun --nnodes=1 \
  --nproc_per_node=1 \
  --standalone run/launcher.py \
  --debug \
  --config <CONFIG_PATH>
```

Additionally, it is possible to override some configs to make debugging easier, for instance:
```bash
torchrun --nnodes=1 \
  --nproc_per_node=1 \
  --standalone run/launcher.py \
  --debug \
  --config <CONFIG_PATH> \
  experiment.torch_compile=false \
  data.train.dataloader.batch_size=64
```

You will need to download non-huggingface datasets separately and implement their respective dataloaders.
We also provide a dataloader for ImageNet.

## General
In ``l3m``, each model is represented as a **MetaModel**, which can be conceptually decomposed into
four (non-mandatory) parts: preprocessor, trunk, postprocessor and head. More concretely, the preprocessor can be
a text-embedding layer or an image patchifier. The trunk can be a transformer or a CNN. The postprocessor can be
normalization or pooling layer. And the head can be a classifier or a projector. Although this may seem limiting,
each part can be composed of multiple blocks (or even full **MetaModels**). The only requirement is the execution order.

<p align="center">
  <img src="docs/_static/metamodel-light.png#gh-light-mode-only" width="700"/>
  <img src="docs/_static/metamodel-dark.png#gh-dark-mode-only" width="700"/>
</p>

At the core, each individual nn.Module is wrapped around a ReadWriteBlock. This adds the basic necessary functionalities
to allow the module access a unified data object, called the data_dict. This makes creating new blocks more flexible,
as they are only expected to take as parameter the data_dict and return it at the end.

<p align="center">
  <img src="docs/_static/readwriteblock-light.png#gh-light-mode-only" width="500"/>
  <img src="docs/_static/readwriteblock-dark.png#gh-dark-mode-only" width="500"/>
</p>

In ``l3m``, all modules have access to the same data_dict, where they can read from and write to. The idea is that
the dictionary is first created by the dataloader and then forwarded through all parts of the model
(including the loss computation). Because of this unified access, the order in which parts of the model are
executed is very flexible and variables can be reused much later in the computational graph.

<p align="center">
  <img src="docs/_static/datadict-light.png#gh-light-mode-only" width="500"/>
  <img src="docs/_static/datadict-dark.png#gh-dark-mode-only" width="500"/>
</p>


### Distributed training
The distributed training in ``l3m`` is done with [FSDP2](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html).
This means that a model will have an underlying [device mesh](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)
that specifies how parameters are stored in memory. For more information regarding parallelization strategies,
please check out these resources: [torchtitan](https://github.com/pytorch/torchtitan),
[Hugging Face's playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) and
[lingua](https://github.com/facebookresearch/lingua).

By default, ``l3m`` will infer the distributed setup following NO_GRAD (``dp_replicate=N_GPUS``, ``dp_shard=1``) or
FULL_GRAD (``dp_replicate=1``, ``dp_shard=N_GPUS``). However, you can specify how the model is sharded.

Model replication (`fsdp/dp_replicate`), sharding (`fsdp/dp_shard`), tensor parallelism (`fsdp/tp_size`) and
context parallelism (`fsdp/cp_size`) are the supported parallelization strategies. The variable names under
parenthesis are used to specify (in the configs) how the model is parallelized.
For example, given a world size of 24 gpus and a config of ``fsdp/dp_replicate=3``, ``fsdp2/dp_shard=4`` and
``fsdp2/tp_size=2``, the model will be replicated across 3 units (each will be a group of 4 * 2 gpus).
Inside each unit the model will be sharded into 4 sub-units (each containing 2 gpus) and then these 2 gpus will work
as a single tensor parallel unit.

## WandB
To use wandb, `.wandb.yaml` must be present in the root of the project containing the following keys:
- `entity` - name of the user/project (e.g. l3m, ...).
- `api-key` - wandb API key.
- `host-name` - machine where the wandb instance is running.


## Citation

### L3M

```bibtex
@misc{apple_l3m,
  author = {Alaaeldin El-Nouby* and Victor Guilherme Turrisi da Costa* and Enrico Fini* and
            Michal Klein* and Mustafa Shukor* and Jason Ramapuram and Jesse Allardic and
            Roman Bachmann and David Mizrahi and Vishnu Banna and Chun-Liang Li and
            Samira Abnar and Vimal Thilak and Joshua M Susskind},
  title = {Large multi-modal models (L3M) pre-training},
  url = {https://github.com/apple/ml-l3m},
  year = {2025}
}
```
\* Core contributors.

### AIMv2
```bibtex
@misc{fini2024multimodal,
    title = {Multimodal Autoregressive Pre-training of Large Vision Encoders},
    author = {Enrico Fini* and Mustafa Shukor* and Xiujun Li and Philipp Dufter and Michal Klein and David Haldimann and
              Sai Aitharaju and Victor Guilherme Turrisi da Costa and Louis Béthune and Zhe Gan and Alexander T Toshev and
              Marcin Eichner and Moin Nabi and Yinfei Yang and Joshua M. Susskind and Alaaeldin El-Nouby*},
    year = {2024},
    eprint = {2411.14402},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

### AIMv1
```bibtex
@InProceedings{pmlr-v235-el-nouby24a,
  title = {Scalable Pre-training of Large Autoregressive Image Models},
  author = {El-Nouby, Alaaeldin and Klein, Michal and Zhai, Shuangfei and Bautista, Miguel \'{A}ngel and
            Shankar, Vaishaal and Toshev, Alexander T and Susskind, Joshua M. and Joulin, Armand},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {12371--12384},
  year = {2024},
}
```

## License
Please check out the repository [LICENSE](LICENSE) before using the provided code and models.
