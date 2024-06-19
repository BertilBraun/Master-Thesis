# Download the project
# git clone https://github.com/BertilBraun/Master-Thesis.git ~/Master-Thesis

if [ ! -d ~/Master-Thesis ]; then
    # exit if the project is not downloaded
    echo "The project is not downloaded. Please download the project first and place it in the home directory."
    exit 1
fi


# If miniconda is not installed, download and install it
if [ ! -d ~/miniconda3 ]; then
    echo "Downloading and installing Miniconda."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
else
    echo "Miniconda is already installed."
fi

# Initialize conda in your bash shell
~/miniconda3/bin/conda init bash
source ~/.bashrc

cd ~/Master-Thesis

# Create the conda environment if it does not exist
if [ ! -d ~/miniconda3/envs/MA ]; then
    echo "Creating conda environment MA."
    conda env create -f src/dpo_cluster/environment.yml
else
    echo "Conda environment MA already exists."
fi

conda activate MA

# Create the workspace if it cannot be found with the ws_find command
if [ "$(ws_find MA)" == "" ]; then
    echo "Creating workspace MA."
    ws_allocate MA 60
else
    echo "Workspace MA already exists."
fi