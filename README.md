# PyTORCS

> This is being developed as part of my Final Year and Industry Exchange project with IBM, titled "**F1 Style Race Replay Commentary with IBM Granite**".

This module is used to extract important telemetry data from a race session in TORCS and output it to a CSV file.

## Installation

This project includes a submodule for [torcs-1.3.7](https://github.com/fmirus/torcs-1.3.7), which has instructions for setup on its repo page and also in [this video](https://www.youtube.com/watch?v=lMzk5HW_kLk).

```bash
git clone https://github.com/DarkSoulWind/pytorcs.git --recurse-submodules
```

Ensure all python modules are installed (TODO)

```bash
cd pytorcs
pip install -r requirements.txt
```

## Running

To run the TORCS build

```bash
cd torcs-1.3.7
./BUILD/bin/torcs
```

To run the PyTORCS client

```bash
python3 -m pytocl.main
```
