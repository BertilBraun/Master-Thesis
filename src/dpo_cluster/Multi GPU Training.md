# Multi GPU Training using Accelerate

<https://huggingface.co/docs/trl/main/en/customization>
<https://huggingface.co/docs/accelerate/en/basic_tutorials/launch>

## Setup

First, install the `accelerate` library:

```bash
pip install accelerate
accelerate config # to create the configuration file
```

To run the training script on multiple GPUs, you can use the `accelerate launch` command. This command will launch the training script on all mentioned GPUs in the configuration file.

```bash
accelerate launch training_script.py
```

## TODO

I think that `accelerate launch` has to be run with a single script, so we will need to merge all the functionality required for training into a single script. The evaluation would therefore be better done in a separate script.
