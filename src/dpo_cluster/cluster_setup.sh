if [ ! -d ~/Master-Thesis ]; then
    # exit if the project is not downloaded
    echo "The project is not downloaded. Please download the project first and place it in the home directory. 'git clone https://github.com/BertilBraun/Master-Thesis.git ~/Master-Thesis'"
    exit 1
fi

cd ~/Master-Thesis

# Add the following to the end ot the .bashrc file
echo "cd ~/Master-Thesis" >> ~/.bashrc
echo "module purge" >> ~/.bashrc
echo "module load compiler/gnu/13.3" >> ~/.bashrc
echo "module load devel/python/3.12.3_gnu_13.3" >> ~/.bashrc
echo "module load devel/cuda/12.2" >> ~/.bashrc
echo "source .venv/bin/activate" >> ~/.bashrc

module purge
module load compiler/gnu/13.3
module load devel/python/3.12.3_gnu_13.3
module load devel/cuda/12.2

if ! ninja --version; then
    echo "Please install ninja. Visit https://pypi.org/project/flash-attn/ for more information."
    exit 1
fi

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

pip install flash-attn --no-build-isolation


# Create the workspace if it cannot be found with the ws_find command
if [ "$(ws_find MA)" == "" ]; then
    echo "Creating workspace MA."
    ws_allocate MA 60
else
    echo "Workspace MA already exists."
fi
