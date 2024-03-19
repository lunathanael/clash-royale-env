# MSU AI Clash Royale RL
Set of tools to generate self-play data using google paly games beta, AutoHotkey, and Windows API for pixel detection.
The application is built with the help of [mctx](https://github.com/google-deepmind/mctx/tree/main), using JIT Compilation of JAX.
Using Python 3.12.1

## Installation
Clone the repository then install the required dependencies.
```bash
git clone https://github.com/lunathanael/clash-royale-env.git
cd clash-royale-env
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
```

## Setup
1. Download the latest google play games from https://play.google.com/googleplaygames.
2. Download and run Clash Royale from the google play games application.
3. If relevant, set your system time to be within 9am - 7pm.

## Quick Start
Run a self-play worker as such

```bash
python -u .\uniform_loop.py --host=true --buffer_size=50 --num_simulations=2
```

Self-play data generation requires two workers to generate data.
