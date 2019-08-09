### Assignment 1: Read the Custom Layers Guide

https://docs.openvinotoolkit.org/2019_R2/_docs_HOWTO_Custom_Layers_Guide.html

`If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.`

Note: to do actual custom layer work: https://github.com/david-drew/OpenVINO-Custom-Layers/tree/master/2019.r2.0

`Layer — The abstract concept of a math function that is selected for a specific purpose (relu, sigmoid, tanh, convolutional). This is one of a sequential series of building blocks within the neural network.`

`Kernel — The implementation of a layer function, in this case, the math programmed (in C++ and Python) to perform the layer operation for target hardware (CPU or GPU).`

`Inference Engine Extension — Device-specific module implementing custom layers (a set of kernels).`

Supported devices: the Intel processors and accelerators, FPGA or Movidius compute sticks.

Alternative to custom layer if device not supported is to have a fallback device.
```
Note: If a device doesn't support a particular layer, an alternative to creating a new custom layer is to target an additional device using the HETERO plugin. The Heterogeneous Plugin may be used to run an inference model on multiple devices allowing the unsupported layers on one device to "fallback" to run on another device (e.g., CPU) that does support those layers.
```

When implementing a custom layer for your pre-trained model in the Intel® Distribution of OpenVINO™ toolkit, you will need to add extensions to **both** the Model Optimizer and the Inference Engine.

The Model Optimizer starts with a library of known extractors and operations for each supported model framework which **must be extended** to use each unknown custom layer. The custom layer extensions needed by the Model Optimizer are:

2 extensions needed for MO: *Custom Layer Extractor* and *Custom Layer Operation* ... why 2?

Extend these supported framework layers. https://docs.openvinotoolkit.org/2019_R1.1/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html .. Standard TensorFlow* operations. About 20 of them. Will need to add another.

For Inference Engine:

```
Custom Layer CPU Extension
A **compiled shared library** (what's this?) (.so or .dll binary) needed by the CPU Plugin for executing the custom layer on the CPU.

Custom Layer GPU Extension
OpenCL source code (.cl) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (.xml) needed by the GPU Plugin for the custom layer kernel.
```

__extension_generator__

python3 /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py new --help
ImportError: No module named 'cogapp'
Shanes-MacBook-Pro-6:demo mm_shane$ pip3 install cogapp
.. now works

Why go through all this work for custom layers? Will the be an important part of the adoption of openVino? Will people give up and run custom tensorflow code on something else?

Step 1: Generate: Use the Model Extension Generator to generate the Custom Layer Template Files.

Step 2: Edit: Edit the Custom Layer Template Files as necessary to create the specialized Custom Layer Extension Source Code.

Step 3: Specify: Specify the custom layer extension locations to be used by the Model Optimizer or Inference Engine.

Why no Keras support?

I am so lost on this registering customer layers on MO part: https://docs.openvinotoolkit.org/2019_R2/_docs_MO_DG_prepare_model_customize_model_optimizer_Extending_Model_Optimizer_with_New_Primitives.html

It's all to get a good IR for hardware.

Problem:
Note https://docs.openvinotoolkit.org/2019_R2/MO_FAQ.html#FAQ1 FAQ1 not found.

Caffe* Models with Custom Layers .. doesn't really have code examples.

Why can't we just create a custom model and easily load it into Intel's IR.... with all the extensions, is it hackable, is someone working on this? This is too much for your average data scientist. Your average data scientist is more concerned with data.

.. I think at the very end was what I would really want:
https://docs.openvinotoolkit.org/2019_R2/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html

it has "Freezing Custom Models in Python*"

Internally, when you run the Model Optimizer, it loads the model, goes through the topology, and tries to find each layer type in a list of known layers. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in this list of known layers, the Model Optimizer classifies them as custom.

I'm not sure about sub-graph: https://docs.openvinotoolkit.org/2019_R2/_docs_MO_DG_prepare_model_customize_model_optimizer_Subgraph_Replacement_Model_Optimizer.html

... doing this: https://github.com/david-drew/OpenVINO-Custom-Layers/blob/master/2019.r2.0/ReadMe.Linux.2019.r2.md ..  i got to:
`mo_tf.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCosh/Activation_8/softmax_output" --extensions $CLWS/cl_cosh/user_mo_extensions --output_dir $CLWS/cl_ext_cosh` .. but am confused. where did i add cosh logic? It seems like lots of copy paste commands, so why is not automated to a higher level extension.

### Assignment 2: Read the Inference Engine Kernels Extensibility Guide

https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Integrate_your_kernels_into_IE.html

`In short, you can plug your own kernel implementations into the Inference Engine and map them to the layers in the original framework.` .. not really sure what this means.

You can find the examples of CPU-targeted kernels in the <INSTALL_DIR>/deployment_tools/inference_engine/src/extension directory. You can also use as an example global GPU kernels delivered with the OpenVINO toolkit.
__c++ code__!

OpenCL C++ kernel language is a static subset of the C++14 standard and includes classes, templates, lambda expressions, function overloads and many other constructs for generic and meta-programming

How to Implement Custom GPU Layers
The GPU codepath abstracts many details about OpenCL™. You need to provide the kernel code in the OpenCL C and the configuration file that connects the kernel and its parameters to the parameters of the layer.

DOESNT tells you what directory to run this from .. dont like path to model.
$ ./classification_sample -m <path_to_model>/bvlc_alexnet_fp16.xml -i ./validation_set/daily/227x227/apron.bmp -d GPU

WHY is this different than doc #1? just GPU?

Currently the Inference Engine Python* API is supported on Ubuntu* 16.04 and 18.04, Windows* 10, macOS* 10.x and CentOS* 7.3 OSes. Supported Python* versions:

### Assignment 3: walk through the custom layer linux tutorial

https://github.com/david-drew/OpenVINO-Custom-Layers/blob/master/2019.r2.0/ReadMe.Linux.2019.r2.md

Edit the CPU Extension Template Files

The CPU extension library is now ready to be compiled. Compile the library using the command:

`make -j $(nproc)` -- nproc not found on mac .. i have 4 cores though


### Notes on Installing the toolkit on MacOS and testing

https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html

add the source to .bash_profile

```
When you run a pre-trained model through the Model Optimizer, your output is an Intermediate Representation (IR) of the network. The IR is a pair of files that describe the whole model:

.xml: Describes the network topology
.bin: Contains the weights and biases binary data

The Inference Engine reads, loads, and infers the IR files, using a common API on the CPU hardware.
```

Installing prereqs for the frameworks like tensorflow, caffe. Hmm .. might need to do this with anaconda3 set as python path.

```
cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites

```

Run the image classification verification steps.

```
cd /opt/intel/openvino/deployment_tools/demo
./demo_squeezenet_download_convert_run.sh
```

Got error:

```
========= Downloading /Users/mm_shane/openvino_models/models/FP16/classification/squeezenet/1.1/caffe/squeezenet1.1.caffemodel
Error Connecting: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /DeepScale/SqueezeNet/raw/a47b6f13d30985279789d08053d37013d67d131b/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel (Caused by SSLError(SSLError(1, '[SSL: TLSV1_ALERT_PROTOCOL_VERSION] tlsv1 alert protocol version (_ssl.c:645)'),))


###############|| Post processing ||###############

FAILED:
squeezenet1.1
Error on or near line 180; exiting with status 1
```

Download manually?
github.com/DeepScale/SqueezeNet/raw/a47b6f13d30985279789d08053d37013d67d131b/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel .. no won't work.

... https://stackoverflow.com/questions/44316292/ssl-sslerror-tlsv1-alert-protocol-version

`pip3 install 'requests[security]'` did the trick!!

Demo completed.

--------------

Trying inference test:

```
brew install cmake
./demo_security_barrier_camera.sh
```

Worked
```
Run Inference Engine security_barrier_camera demo

Run ./security_barrier_camera_demo -d CPU -d_va CPU -d_lpr CPU -i /opt/intel/openvino/deployment_tools/demo/car_1.bmp -m /Users/mm_shane/openvino_models/ir/FP16/Security/object_detection/barrier/0106/dldt/FP16/vehicle-license-plate-detection-barrier-0106.xml -m_va /Users/mm_shane/openvino_models/ir/FP16/Security/object_attributes/vehicle/resnet10_update_1/dldt/FP16/vehicle-attributes-recognition-barrier-0039.xml -m_lpr /Users/mm_shane/openvino_models/ir/FP16/Security/optical_character_recognition/license_plate/dldt/FP16/license-plate-recognition-barrier-0001.xml

[ INFO ] InferenceEngine:
	API version ............ 2.0
	Build .................. 27579
	Description ....... API
[ INFO ] Files were added: 1
[ INFO ]     /opt/intel/openvino/deployment_tools/demo/car_1.bmp
[ INFO ] Loading device CPU
	CPU
	MKLDNNPlugin version ......... 2.0
	Build ........... 27579

[ INFO ] Loading detection model to the CPU plugin
[ INFO ] Loading Vehicle Attribs model to the CPU plugin
[ INFO ] Loading Licence Plate Recognition (LPR) model to the CPU plugin
[ INFO ] Number of InferRequests: 1
[ INFO ] Display resolution: 1920x1080
[ INFO ] Number of allocated frames: 3
[ INFO ] Resizable input with support of ROI crop and auto resize is disabled

Mean overall time per all inputs: 362225.05ms / 0.00FPS for 1 frames
Detection InferRequests usage: 100.00%
[ INFO ] Execution successful


###################################################

Demo completed successfully.
```

--------------

Benchmark Demo

```
Count:      1000 iterations
Duration:   5757.19 ms
Latency:    22.3672 ms
Throughput: 173.696 FPS
```

### About Intel AI

https://www.forbes.com/sites/janakirammsv/2019/05/26/running-deep-learning-models-on-intel-hardware-its-time-to-consider-a-different-os/#11cf1ade1f4e

Firstly, Intel has done extensive work to make the Xeon family of processors highly optimized for AI. The Intel Xeon Scalable processors **outsmart GPUs** in accelerating the training on large datasets.

Intel is telling its customers that they don’t need expensive GPUs until they meet a threshold. Most of the deep learning training can be effectively done on CPUs that cost a fraction of their GPU counterparts.

https://software.intel.com/en-us/iot/cloud-analytics/aws

https://medium.com/sugarkubes/guys-you-dont-always-need-a-gpu-openvino-b0739a7d4411

AWS image w/ openVino run on cloud? ... or they want all on people buying new hardware.
https://github.com/sugarkubes/openvino-base-image

### Terminology refresher

Batch Size
Total number of training examples present in a single batch.

Note: Batch size and number of batches are two different things.

But What is a Batch?
As I said, you can’t pass the entire dataset into the neural net at once. So, you divide dataset into Number of Batches or sets or parts.

Just like you divide a big article into multiple sets/batches/parts like Introduction, Gradient descent, Epoch, Batch size and Iterations which makes it easy to read the entire article for the reader and understand it. 😄

Iterations
To get the iterations you just need to know multiplication tables or have a calculator. 😃

Iterations is the number of batches needed to complete one epoch.

Note: The number of batches is equal to number of iterations for one epoch.

Let’s say we have 2000 training examples that we are going to use .

We can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch.

Where Batch Size is 500 and Iterations is 4, for 1 complete epoch.

--------

__weights__ - also known as kernels, filters, templates, or feature extractors

__blob__ - also known as tensor - an N dimensional data structure, that is, an N-D tensor, that contains data, gradients, or weights (including biases)

__units__ - also known as neurons - performs a non-linear transformation on a data blob

__feature maps__ - also known as channels - Yes, both are same. Each channel after the first layer of a CNN is a feature map. Before the first layer of CNN, RGB images have 3 channels (red, green & blue channels). A feature map is the output of the mathematical operation called convolution, whereas images might have multiple channels. Since both are kinda true for CNNs you hear both of the terms floating around.

I would argue calling the output feature maps is probably a little more accurate.

__testing__ - also known as inference, classification, scoring, or deployment

__model__ - also known as topology or architecture
