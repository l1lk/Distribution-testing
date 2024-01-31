# DistXplore-demo
The code demo of DistXplore



## Installation

We have tested DistXplore based on Python 3.7 on Ubuntu 20.04, theoretically it should also work on other operating systems. To get the dependencies of DistXplore, it is sufficient to run the following command.

`pip install -r requirements.txt`

The version of the library 'protobuf' maybe unsuitable, you can run the following command to fix it.

`pip install --upgrade protobuf==3.20`


### Usage

### Distribution-aware testing

We provide a script to generate distribution-aware test samples for LeNet4 model trained on MNIST dataset. You can download other models from the google drive mentioned above.
test
```
test
cd DistXplore/dist-guided
sh generate_demo.sh
```

