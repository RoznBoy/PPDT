# PPDT
Implementing Binary Decision Tree using Fully Homomorphic Encryption

# Paper
You can find the paper at the following link. : <https://eprint.iacr.org/2024/529>  
This paper introduces a new method for training decision trees and random forests using CKKS homomorphic encryption (HE) in cloud environments,  enhancing data privacy from multiple sources.  
The innovative Homomorphic Binary Decision Tree (HBDT) method utilizes a modified Gini Impurity index (MGI) for node splitting in encrypted data scenarios. Notably, the proposed training approach operates in a single cloud security domain without the need for decryption, addressing key challenges in privacy-preserving machine learning.

# Code
The code utilizing HEaaN for GPU acceleration cannot be disclosed.   
Therefore, I will upload a part of the code utilizing pi-heaan and heaan-it, which can be shared.

# Environment setup for heaan-it
- make image
  * sudo docker pull cryptolabinc/heaan-stat:0.2.0-cpu-x86_64-avx512
- make container
  * sudo docker run -it --name <container name> cryptolabinc/heaan-stat:0.2.0-cpu-x86_64-avx512
  * When you enter the above command, Jupyter will be launched.
  * After forcibly terminating with ctrl+c, execute the following command to run Docker.
- docker start
  * sudo docker start <container name>

# Data
![image](https://github.com/RoznBoy/PPDT/assets/154126402/1330d7ef-027c-4887-9df8-5c1a44cb5d0a)
