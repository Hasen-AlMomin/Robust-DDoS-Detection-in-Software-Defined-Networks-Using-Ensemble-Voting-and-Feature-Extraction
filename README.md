# Robust DDoS Detection in Software Defined Networks using Ensemble Voting and Feature Extraction

## Project Description
This project provides a basic framework for **DDoS detection** using **Ensemble Voting** in a **Software Defined Network (SDN)**. The network is implemented using **Mininet** and controlled by **RYU**. The detection system uses a hybrid of classification algorithms for DDoS attack detection.

## Prerequisites
Before you begin, ensure you have the following dependencies installed:

### Install Ubuntu OS
Ensure you are running Ubuntu as your operating system. This guide assumes you are using Ubuntu 20.04 or later.

### Install Mininet, OpenVSwitch, and Ryu
To run the SDN network with Ryu controller and Mininet, follow these installation steps.

1. **Install Mininet**
    ```bash
    sudo apt-get install mininet
    ```

2. **Install OpenVSwitch**
    ```bash
    sudo apt-get install openvswitch-switch
    ```

3. **Install Ryu SDN Controller**
    ```bash
    sudo apt-get install python3-ryu
    ```

4. **Install Python and Dependencies**
    Ensure Python 3.x and pip are installed, then install the required Python libraries:
    ```bash
    sudo apt-get install python3-pip
    pip install numpy scikit-learn pandas matplotlib
    ```

## Running the SDN Environment

### Step 1: Run the Ryu Controller
Navigate to the directory where your Ryu controller application (`app1.py`) is located, and run it using the following command:

```bash
cd /path/to/your/ryu/app/
ryu-manager app1.py
