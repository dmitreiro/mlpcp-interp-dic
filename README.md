# mlpcp-interp
**Machine Learning** for **Prediction** of **Constitutive Parameters** - **Interpolation** study

## :gear: Setup

### Clone

Open terminal, change your current working directory to the location where you want the cloned directory and then clone this repository to your local machine

```
git clone https://github.com/dmitreiro/mlpcp-interp.git
```

### Config

Inside your repository home folder, edit ```config/config.ini``` file to define your variables.

### Environment

Next, install **Anaconda** for managing your Python environments. You can check documentation [here](https://docs.anaconda.com/anaconda/install/).\
After the installation, create an empty environment using **Python 3.11.10**

```
conda create --name <your_env_name> python=3.11.10
conda activate <your_env_name>
```

Then, navigate to your repository home folder and install dependencies

```
pip install -r requirements.txt
```

### Run your code

Now, you are ready to rock :sunglasses:\
From the home folder, just run

```
python main.py
```

or, if you want to run a specific script

```
python <folder>/<script>.py
```


## :balance_scale: License

This project is licensed under the MIT License, which allows anyone to use, modify, and distribute this software for free, as long as the original copyright and license notice are included. See the [LICENSE](LICENSE) file for more details.
