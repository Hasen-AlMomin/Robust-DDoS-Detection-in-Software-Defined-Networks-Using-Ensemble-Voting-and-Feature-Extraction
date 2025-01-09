# Robust-DDoS-Detection-in-Software-Defined-Networks-Using-Ensemble-Voting-and-Feature-Extraction

---------------------------------------------------------------
Getting Started
Clone the repository

git clone https://github.com/Hasen-AlMomin/Robust-DDoS-Detection-in-Software-Defined-Networks-Using-Ensemble-Voting-and-Feature-Extraction


## Project Description
This project provides a basic framework for **DDoS detection** using **Ensemble Voting** in a **Software Defined Network (SDN)**. The network is implemented using **Mininet** and controlled by **RYU**. The detection system uses a hybrid of classification algorithms for DDoS attack detection.

## Prerequisites
Before you begin, ensure you have the following dependencies installed:

### Install Ubuntu OS 18.04
Ensure you are running Ubuntu as your operating system. This guide assumes you are using Ubuntu 18.04.

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

Step 2: Run the Network Topology
cd /path/to/your/mininet/topo/
sudo python3 topo5.py

Step 3: Test the Environment (DDoS Attack Simulation)
sudo apt-get install hping3
Then, run the DDoS attack with the following command:
sudo hping3 --flood -p 80 <target-ip>


Step 4: Running the Detection Algorithms

1.Run the Enesemble_voting algorithm accuracy score:
python accuracy_score_Hybrid.py

2.Generate the confusion matrix:
python confusion_matrix.py

3.Evaluate the detection rate:
python detection_rate.py




Prerequisites
Install dependencies
Install Mininet
Install OpenVSwitch
Install Ryu
Install UBUNTU 
Install Python

run RYU controller navigate to the path and run a command 
ryu-manager app1.py

Run the Topology 
sudo python3 topo5.py

Test the Enviroment By using Hping3 as a DDoS attack 

To run Analysis Algorithms 
python accuracy_score_Hybird.py
confusion_matrix.py
detection_rate.py
## License
This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

