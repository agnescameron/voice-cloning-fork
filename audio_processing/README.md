# audio sample processing

Command line script that gets a youtube video, downloads it, and converts it to a 16-bit .wav file for use in the toolbox (has to be 16 bit cause python doesn't like it any other way).

To use, pass the url of the youtube video as the first argument, what directory you want it in as the second, and what you want to name the file as the third. It'll output `filename.wav` in the parent directory.

```
$ ./audio_processing.sh [<youtube-url>] [<directory>] [<filename>]
```

This script doesn't trim the audio: I use Audacity for longer clips / clips with two voices, in general, 5 or 6 clips of around 30 seconds in length get reasonable results with the model. 
