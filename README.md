# AuxiliaryASR
This repo is used as a Norweagian Phoneme-level ASR in TTS as a text aligner for StyleTTS2 ([https://github.com/yl4579/StyleTTS2])
Can be used for other lang, but i used it for Norweagian.
This repo is based on https://github.com/yl4579/AuxiliaryASR



## Current Edits compared to the original repo
- Switched from CNN to Conformer blocks
- Added whisper to use transfer learning, but not currently using it
- Added so that you get GT and predicted sequence is given in wandb with WER score
- to get the whisper to work you need to uncomment the blocks in the trainer and in the meldataset.py you need to return the wav
- The `train_list.txt` and `val_list.txt` expects it to already be phonemized




---


## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/yl4579/AuxiliaryASR.git
cd AuxiliaryASR-NB
```
3. Install python requirements: 
```bash
pip install SoundFile torchaudio torch jiwer pyyaml click matplotlib g2p_en librosa
```
4. Prepare your own dataset and put the `train_list.txt` and `val_list.txt` in the `Data` folder (see Training section for more details).

## Training
```bash
python train.py --config_path ./Configs/config.yml
```
Please specify the training and validation data in `config.yml` file. The data list format needs to be `filename.wav|label|speaker_number`, see [train_list.txt](https://github.com/yl4579/AuxiliaryASR/blob/main/Data/train_list.txt) as an example (a subset for LJSpeech). Note that `speaker_number` can just be `0` for ASR, but it is useful to set a meaningful number for TTS training (if you need to use this repo for StyleTTS). 

Checkpoints and Tensorboard logs will be saved at `log_dir`. To speed up training, you may want to make `batch_size` as large as your GPU RAM can take. However, please note that `batch_size = 64` will take around 10G GPU RAM. 

### Languages
This repo is set up for English with the [g2p_en](https://github.com/Kyubyong/g2p) package, but you can train it with other languages. If you would like to train for datasets in different languages, you will need to modify the [meldataset.py](https://github.com/yl4579/AuxiliaryASR/blob/main/meldataset.py#L86-L93) file (L86-93) with your own phonemizer. You also need to change the vocabulary file ([word_index_dict.txt](https://github.com/yl4579/AuxiliaryASR/blob/main/word_index_dict.txt)) and change `n_token` in `config.yml` to reflect the number of tokens. A recommended phonemizer for other languages is [phonemizer](https://github.com/bootphon/phonemizer).

## References
- [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)

## Acknowledgement
The author would like to thank [@tosaka-m](https://github.com/tosaka-m) for his great repository and valuable discussions.
