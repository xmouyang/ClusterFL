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
