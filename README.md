# ClusterFL
This is the repo for MobiSys 2021 paper: "ClusterFL: A Similarity-Aware Federated Learning System for Human Activity Recognition".
<br>

# Requirements
The program has been tested in the following environment: 
* Ubuntu 18.04
* Python 3.6.8
* Tensorflow 2.4.0
* sklearn 0.23.1
* opencv-python 4.2.0
* Keras-python 2.3.1
* numpy 1.19.5
<br>

# ClusterFL Overview
<p align="center" >
	<img src="./figures/ClusterFL-system-overview.pdf" width="1000">
</p>

# Project Strcuture
```
|-- client                    // code in client side
    |-- client_cfmtl.py/	// main file of client 
    |-- communication.py/	// set up communication with server
    |-- data_pre.py/		// prepare for the FL data
    |-- model_alex_full.py/ 	// model on client 
    |-- desk_run_test.sh/	// run client 

|-- server/    // code in server side
    |-- server_cfmtl.py/        // main file of client
    |-- server_model_alex_full.py/ // model on server 

|-- README.md

|-- pictures               // figures used this README.md
```
<br>

# Run
## Quick Start
* Download the `dataset` folders (collected by ourself) from [FL-Datasets-for-HAR](https://github.com/xmouyang/FL-Datasets-for-HAR) 
* Change the "read-path" in 'data_pre.py' to the folder of above dataset on your client machine.
* Change the "server_addr" and "server_port" in 'client_cfmtl.py' as your true server address. 
* Run the following code on the client machine
    ```bash
    cd client
    ./desk_run_test.sh
    ```
* Run the following code on the server machine
    ```bash
    cd server
    python3 server_cfmtl.py
    ```
    ---

# Citation
If you find this work or the datasets useful for your research, please cite:
    ```bash
    @inproceedings{ouyang2021clusterfl,
    title={ClusterFL: a similarity-aware federated learning system for human activity recognition},
    author={Ouyang, Xiaomin and Xie, Zhiyuan and Zhou, Jiayu and Huang, Jianwei and Xing, Guoliang},
    booktitle={Proceedings of the 19th Annual International Conference on Mobile Systems, Applications, and Services},
    pages={54--66},
    year={2021}
    }
    ```
    ---
    
