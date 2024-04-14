```bash
#!/bin/bash

# Install system-level dependencies
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    brew install pkg-config cairo
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux (Generic)
    sudo apt-get install pkg-config libcairo2-dev  # Adjust as necessary for your distro
fi

# Install Python dependencies
pip install -r requirements.txt