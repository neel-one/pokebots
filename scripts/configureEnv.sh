# ideally this script works on linux instantly, but likely not lol

# sanity check
alias python=python3

# Configure venv
if [ ! -d "/env" ] 
then
    echo "Building virtual environment" 
    python3 -m venv env
    source env/bin/activate
fi

# Build dependencies
pip3 install -e .
pip3 install -r requirements.txt


