# Real-Time Voice Cloning
This repository is a fork of CoretinJ's voice cloning repo.

**Video demonstration** (this is a pretty good visual overview of the toolbox):

[![Toolbox demo](https://i.imgur.com/8lFUlgz.png)](https://www.youtube.com/watch?v=-O_hYhToKoA)

## Quick start
### Requirements
In order to use the repository, you'll need:

**Python 3.7**. Python 3.6 might work too, but I wouldn't go lower because I make extensive use of pathlib.

Run `pip install -r requirements.txt` to install the necessary packages. Additionally you will need [PyTorch](https://pytorch.org/get-started/locally/) (>=1.0.1).

### Pretrained models
Download the latest [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models). These follow the same file structure as the repository: so, copy `saved_models` in the folder `encoder` into the folder `encoder` in the repo, etc.

### Output
I added an 'output' line to the vocoder. This will dump files in `output_files`, named with the unix timestamp and the parent directory.

### Toolbox
To run the toolbox on a mac, do

`python demo_toolbox.py --low_mem`  

The `--low_mem` tag is important for running on a CPU (I tried without, and couldn't get any sound out). If you have the error `Aborted (core dumped)`, see [this issue](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/11#issuecomment-504733590).

This should open up an interface like the one in the video: browse to import recorded audio, or record your own.


### Datasets
CoretinJ suggests the following datasets, if you want to use a set of pre-recorded voices: 
For playing with the toolbox alone, I only recommend downloading [`LibriSpeech/train-clean-100`](http://www.openslr.org/resources/12/train-clean-100.tar.gz). Extract the contents as `<datasets_root>/LibriSpeech/train-clean-100` where `<datasets_root>` is a directory of your choosing. Other datasets are supported in the toolbox, see [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training#datasets). You're free not to download any dataset, but then you will need your own data as audio files or you will have to record it with the toolbox.

To start the toolbox with a pre-loaded dataset, do `python demo_toolbox.py -d <datasets_root>`. The tool complains if you try and point it at a data set it doesn't know, but there's an option to load external files into the GUI, which is easier.

## Wiki
- **How it all works** (WIP - [stub](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/How-it-all-works), you might be better off reading my thesis until it's done)
- [**Training models yourself**](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Training)
- **Training with other data/languages** (WIP - see [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/30#issuecomment-507864097) for now)
- [**TODO and planned features**](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/TODO-&-planned-features) 

