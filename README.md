# 🌵 **simplellm**

🌵 **simplellm** is a (in-progress) library for developing and training Transformer Language Models
with the goal of being as simple as possible. A repository of my quest to build high performing models
and training infrastructure as simply as possible.

## Project Struture
🌵 simplellm is split into sub-directories: ```modeling``` and ```training```.

### Modeling
Model building blocks and their supporting code live in the ```modeling``` directory. Components can exist
at multiple levels of abstraction; from operations like ```ALiBiAttention``` to the ```PerceiverBlock```.
This is to facilitate maximal reuse across projects by capturing components defined at varying levels
of abstraction. Fused kernels and quantized implementations will also find their way into this directory.

### Training
The ```training``` directory contains code for a ```Trainer``` object which defines an event loop surrounding
the training process. Similar to fastai, PyTorch Lightning, and MosaicML Composer, the 🌵 simplellm
```Trainer``` utilizes externally defined callback functions which are invoked at specified points in the
lifecycle of the training process. The ```Trainer``` object will also support DDP and FSDP training.