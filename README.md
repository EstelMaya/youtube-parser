# YouTube video parser

This project allows you to produce speech datasets from subbed YouTube videos.

It checks whether provided video has subtitles in required language, downloads subs and audio from video, diarises audio and outputs results either as text-audio pairs, or produces metadata file containing time intervals for each given speaker, and optionally removes noise segments.

Examples can be found in `examples.py`.


### Acknowledgement
Diarization code is taken from https://github.com/cvqluu/simple_diarizer
