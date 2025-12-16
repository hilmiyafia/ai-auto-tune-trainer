# AI Auto Tune Trainer
This repository contains the OpenUtau AI Auto Tune trainer.

## How to Use
See this video for the details: [video](https://youtu.be/wphtRVZiujs?t=316)
1. Gather your audio samples.
2. Label your audio samples. The labels should be separated by notes, NOT by phonemes like in DiffSinger.
3. Convert the labels to USTs. Make sure that the notes in the USTs are on the correct pitch (not flat). And make sure that each UST has the same file name to its corresponding audio sample.
4. Put the audio samples in the audios folder, and the USTs in the usts folder.
5. Run run_cuda.bat or run_cpu.bat, according to your computer.
6. If you want to pause the training, you can close the training window at anytime. Later, just run either the run_cuda.bat or the run_cpu.bat again and it will automatically resume your training from the latest checkpoint.
7. You can check the config.py file to change the training parameters.
8. After the training is finished, a pitch.onnx file will be generated in the folder. Move this model to your voicebank folder, or the plugins folder inside OpenUtau to replace the default fallback pitch model.
