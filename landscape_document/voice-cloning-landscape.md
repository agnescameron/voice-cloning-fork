---
header-includes:
  - \hypersetup{colorlinks=false,
            allbordercolors={0 0 0},
            pdfborderstyle={/S/U/W 1}}
---


# Voice Cloning Landscape Documentation

## Technical Background
Deep-learning based voice synthesis models broadly operate on the principle I described in the office: you have some representation of 'lots of voices' (that you get from 'training' your model), *U*, and some voice, *V*. Both *U* and *V* are represented by vectors, where each element of the vector represents some 'feature' of the voice. These vectors of features are known as *embeddings*. Much of the work that goes into making speech synthesisers is in determining which features the encoder will extract.

There's a number of different techniques used to create these embeddings, and what the model then does with the embedding also depends on how the synthesiser works.

![Diagram of processes for training, cloning and generation, based on either adapting a new speaker to some model (left), or creating a voice model based on a single speaker (right). The method that's probably most useful for this project is second one down on the left: an existing, trained universal speaker model, and a custom-generated speaker embedding.](voice-cloning.png "Voice Cloning Diagram")

### Autoregressive Models
Autoregressive models use observations from previous time-steps as inputs to predict what will happen in the next time-step. As audio data varies with time, this can be used to model speech in terms of linguistic features.

The problem with autoregressive models in audio is that raw audio generates a vast number of data points every second: to consider all of these in predicting the next step of, say, a phoneme is extremely resource-heavy.

### GANs
These are the models used to make the [uncanny images](https://futurism.com/incredibly-realistic-faces-generated-neural-network) -- and are generally a neat way to dump a lot of data in, and get some pretty convincing approximations out.

The basic idea is that you have one neural network that attempts to make, say, faces (based on a bunch of images of faces you have as input), and another that judges how 'face-like' the 'faces' produced by the first network are. The classifier data from the second network is fed back to the first, and the two train each other.

[Until recently](https://arxiv.org/pdf/1910.06711.pdf) these haven't been a great way to produce audio (partly because of uncertainty in what feature of the audio should be extracted to produce a realistic synthesis), but the MelGAN/Lyrebird people seem to have improved this.

### Mel Spectrograms
These are pretty common bases for all the models, and are based on the Mel (melody) scale, a scale of pitches judged by listeners to be equal in distance from one another (these get much bigger as you reach higher pitches). 

As they're based on human perception, Mel-scale spectrograms are often used to represent things like speech and music, as they trade off detail at higher frequencies which we can't hear anyway for much greater detail in key human-audible range. (in signal processing this is called [companding](https://en.wikipedia.org/wiki/Companding)).

### Improving results
Tricks to get better results vary with the model used to synthesise speech. In general, having more data samples, particularly data with different kinds of background noise (so that the model doesn't think that the noise is part of the speech) is useful for getting better results.

\newpage

## Significant Research

### [WaveNet, 2016](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
WaveNet's main innovation was a number of tricks that take what would be an almost impossibly slow task (a fully autoregressive model trained on raw audio) down to nearly realtime performance. They use something called [Dilated Convolution](https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5), which artifically expands the size of the 'input field' that gets fed into the model, allowing it to process thousands of timesteps simultaneously.

Why is raw audio good? It picks up on non-speech sounds (breathing, pauses) that aren't easily modelled by feature extractors that reduce the dimensions of the input. At the time, this model was a really big deal, and basically totally changed the field. The name comes from modelling 'waveforms', e.g. directly from spectrograms. 

### [Tacotron, 2017](https://arxiv.org/abs/1703.10135)
A Google and Facebook model for an end-to-end Text-To-Speech model that generates acoustic features from input characters. Before this, TTS required a number of different components (e.g. text analysis).

### [Tacotron-2, 2018](https://github.com/Rayhane-mamah/Tacotron-2)
This is the combination of the WaveNet vocoder with a revised version of the Tacotron architecture. Makes big improvementson the original.

It (and similar model [Char2Wav](http://www.josesotelo.com/speechsynthesis/)) works by modelling a low-resolution version of the input (a 'reader' stage), and then using that version as the input for a high-resolution output (the 'neural vocoder'), which adds in a lot of the more natural linguistic features.

The [github repo](https://github.com/Rayhane-mamah/Tacotron-2) claims that a pre-trained model will be added at a later date, but as yet nothing has been put online.

### [MelGAN, 2019](https://arxiv.org/abs/1910.06711)
First real high-quality audio generation with a GAN (rather than an autoregressove network), used in [Lyrebird's](#Lyrebird-AI) voice synthesis service.

### [FastSpeech (2019)](https://arxiv.org/abs/1905.09263)
Microsoft's attempt at entering the speech synthesis party. A non-autoregressive Text-to-Speech model (so not wading through lots of data), seems potentially interesting. There's a nice blog post about it [here](https://www.microsoft.com/en-us/research/blog/fastspeech-new-text-to-speech-model-improves-on-speed-accuracy-and-controllability/)

\newpage

## Open Source Implementations
In terms of completeness for end-to-end voice cloning purposes, the best bet is probably the [Coretin-J](https://github.com/CorentinJ/Real-Time-Voice-Cloning) repository. Most of the other open-source elements aren't end-to-end (e.g. are just the feature extraction or text-to-speech stages), and also require you to train them yourself. 

### [Coretin-J Model](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
This is based on [this paper](https://arxiv.org/pdf/1806.04558.pdf) that came out of Google in 2019, which sticks together 3 pre-existing components to synthesise natural speech from speakers dissimilar from those used in training.

* encoder -- this derives the embedding of the voice sample that you give it. they train a network on thousands of speakers, which allows the encoder to create an embedding vector from a very small amount of input data
* synthesiser -- when conditioned using a speaker's embedding, generates a spectrogram from text (this is based on Tacotron-2)
* vocoder -- generates an audio waveform from a generated spectrogram (this is taken from Wavenet)

Each of these components are trained on a large amount of data, which allows them to produce the model that the embedding, spectrogram and waveform may be compared to. These components can be downloaded [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models).

From the demo videos, the repo includes a straightforward Python interface, though the tests imply it requires a CuDA-capable GPU to run (most macs don't have Nvidia GPUs, though [this](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) implies that *some* might). There is, however [a fork](https://github.com/shawwn/Real-Time-Voice-Cloning) to allow it to run on CPU: probably totally acceptable for using the pre-trained model, definitely inadequate for training it yourself.

### [NVIDIA Waveglow](https://nv-adlr.github.io/WaveGlow)
This is a non-autoregressive model, based on WaveNet. It's not so clear from the documentation, but I believe it's possible to use this model to clone your own audio samples, provided they're in the correct format (see below). However, given that this is a NVIDIA model, it will almost certainly require an NVIDIA GPU (not in Macs), and I can't see anything on the forks page that references CPUs.

They *do* have a pre-trained model, which can be downloaded [here](https://drive.google.com/file/d/1cjKPHbtAMh_4HTHmuIGNkbOkPBD9qwhj/view). However, instead of running the file with an mp3 voice sample, the input must be a mel-spectrogram. However, that shouldn't be too difficult to make from raw audio (there's a [MatLab library for it](https://www.mathworks.com/help/audio/ref/melspectrogram.html)). It's also unclear how *much* audio you need from a particular speaker, though this could maybe be determined by going through their example mel-spectrograms.

### [Deep Voice Conversion](https://github.com/andabi/deep-voice-conversion)
A style transfer model that uses a phoneme classifier, then a synthesiser to process input audio files to build a model of a speaker. Like Coretin J's model, the input data doesn't need to be labelled, however, it requires a lot more of it (2 hours at least).

The demo sample of style transfer of Kate Winslet is [pretty crunchy](https://soundcloud.com/andabi/sets/voice-style-transfer-to-kate-winslet-with-deep-neural-networks), and at the moment I don't know that this repo is worth the effort.

### [Neural Voice Cloning with Few Samples](https://github.com/SforAiDl/Neural-Voice-Cloning-With-Few-Samples)
An implementation of the [Baidu paper](https://arxiv.org/pdf/1802.06006) with the same name, this claims to implement the model described in the paper. At present, it seems to have some bugs (from the issue tracker), it's not as well documented as the Coretin-J repo, and the [link to the samples](http://saidl.in/Neural-Voice-Cloning-With-Few-Samples/) is down, so hard to know if it's any good.

### [MelGAN](https://github.com/descriptinc/melgan-neurips)
Open-sourced version of the model used in [Lyrebird's](#Lyrebird-AI) voice synthesis service. Does not include a pre-trained model, so would be a lot of work to put this to use.

\newpage

## Proprietary Platforms
In general (and, I guess, as one might hope) all of the high quality voice cloning platforms that I've come across claim that they only allow you to clone *your* voice. Some, like Lyrebird, who are still in Beta might be happy to let you clone Welles/Wilde etc if you send them an email.

### [Resemblyzer](https://www.resemble.ai)
The proprietary platform developed from the Coretin repo, high quality (imo not as high quality as Lyrebird), with prices starting at $50/month to process audio. Allows the creation of custom voices, but limited to the speaker's own. *Not* in Beta, as compared to Lyrebird -- if it's possible to get around the mic issue (and if the Coretin repo proves too annoying to get to work) this might be the most convenient way to produce the audio.

### [Lyrebird AI](https://www.descript.com/lyrebird-ai?source=lyrebird) 
The quality of this model is really great. It's currently in Beta but you can email them to ask about testing their software. They might well be up for it: they describe themselves as *"using artificial intelligence to enable creative expression"*, so possibly pretty happy to let you test their platform for an art piece. Founded by ex-PhD students of [Yoshua Bengio](https://en.wikipedia.org/wiki/Yoshua_Bengio) (big deep learning guy, just won Turing award), so probably the real deal in terms of technical capabilities.

They use a generative network called [MelGAN](https://arxiv.org/pdf/1910.06711.pdf), a Generative Adversarial Network. Because it's not an autoregressive model, it's naturally faster than WaveNet, and pretty neat. This paper is super recent, and seems to have been authored as part of their PhDs. This [blog post](https://www.descript.com/post/ultra-fast-audio-synthesis-with-melgan) has a nice summary of the paper.

Lyrebird is part of [Descript](https://www.descript.com), a content-aware audio editing platform that seems to be primarily aimed at podcasters. Descript is free for up to 3h/audio per month, and $10/month for unlimited, though this pricing doesn't seem to yet apply to Lyrebird (as it's still in Beta). Given it's already an audio editing platform, I'd also imagine that you'd be able to improve the quality quite a bit using their tools post-generation.

One potential snag is that they say on their [ethics page](https://www.descript.com/ethics) that they only let you clone your own voice -- though you might be able to persuade them otherwise.

### [WellSaid Labs](https://wellsaidlabs.com)
There's [a video](https://www.youtube.com/watch?v=akc1Ddt7rX4) of WSLTTS (well said labs TTS) vs WaveNet, demonstrating a far more 'human-quality' speech (mostly in terms of sentence structure and intonation). This might be cherry-picked, but it's still impressive. Here's a [few more of their voices](https://www.youtube.com/watch?v=evmvsviHNYY). They seem to cater to mostly an e-learning and customer service market. There's every little documentation of the models they use online, so it's not clear how they're getting these results, and their [public-facing git repos](https://github.com/wellsaid-labs) are forks of Phonemizer, and WaveNet implementations.

These guys are also in Beta, and while you can request a free demo of one of their existing voices, it's unclear whether that offer is extended to the 'custom voice feature'. They say that *"WellSaid Labs can design a unique voice tailored to the needs of your project"*, which sounds potentially expensive, but perhaps [worth enquiring](sales@wellsaidlabs.com).

### [adobe VoCo](https://en.wikipedia.org/wiki/Adobe_Voco)
Adobe Audio Maniuplator, [as yet doesn't exist commercially](https://community.adobe.com/t5/video-lounge/is-adobe-voco-dead-news-2018/td-p/9684886). Requires around 20 minutes of speech to produce a voice clone.

[The Demo Video](https://www.youtube.com/watch?v=I3l4XLZ59iw) is kind of wild, the guy demoing fully admits it'll be used for deepfakes: 'you use Photoshop to do weird things online, now you can use this to change audio'.

### [iSpeech](http://www.ispeech.org/)
The product they have online isn't super high-quality. They do have a TTS API, and it appears that they allow you to use [custom voices](http://www.ispeech.org/api/#voices-custom), though it's not clear from the documentation how these 'custom voices' are trained or integrated into the model.

There's also a ['voice cloning'](https://www.ispeech.org/voice-cloning) section on their website, which doesn't give a huge amount of details of what they allow you to do (e.g. it has some cloned samples of famous peoples' voices, and says you can train using *your* voice), but nothing about training other voices (though nothing saying you can't).

Pricing-wise, [there's no free tier](https://www.ispeech.org/developer/purchase/plans), starts at $0.025/word for the API. Would say this looks like a pretty poor option.

### [Cerevoice](https://www.cereproc.com/en/cerevoice-me)
This is very pricey (around $650), and seems to be a more trad model that uses a large amount of high-quality audio to produce a voice model. Also requires it to be mic input. 

\newpage


## Corpuses

* [*Speech Accent Archive*](https://www.kaggle.com/rtatman/speech-accent-archive): The speech accent archive is established to uniformly exhibit a large set of speech accents from a variety of language backgrounds. This dataset contains 2140 speech samples, each from a different talker reading the same reading passage. Talkers come from 177 countries and have 214 different native languages. Each talker is speaking in English. [This website](http://accent.gmu.edu) has more details about the archive.

* [*Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio): The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

* [*Google Research AudioSet*](https://research.google.com/audioset/dataset/index.html): The AudioSet dataset is a large-scale collection of human-labeled 10-second sound clips drawn from YouTube videos. There's also the [Audioset ontology](https://research.google.com/audioset/ontology/index.html), a collection of labelled 'sound events'.

* [*VoxCeleb*](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/): database of celebrity voices

* [*Common Voice*](https://www.kaggle.com/mozillaorg/common-voice): Common Voice is a corpus of speech data read by users on the Common Voice website (http://voice.mozilla.org/), and based upon text from a number of public domain sources like user submitted blog posts, old books, movies, and other public speech corpora. Labels by age, gender, accent.

* [*English Multi-speaker Corpus for CSTR Voice Cloning Toolkit*](https://www.kaggle.com/mfekadu/english-multispeaker-corpus-for-voice-cloning): This CSTR VCTK Corpus includes speech data uttered by 109 native speakers of English with various accents. Each speaker reads out about 400 sentences, most of which were selected from a newspaper plus the Rainbow Passage and an elicitation paragraph intended to identify the speaker's accent. 

* [*CHiME-5*](http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME5/overview.html): A dataset that targets the problem of distant microphone conversational speech recognition in everyday home environments. Comprises 20 recordings of dinner parties.

* [*TIMIT Acoustic-Phoenetic Continuous Speech*](https://catalog.ldc.upenn.edu/LDC93S1): A dataset of speech data for acoustic-phonetic studies, which provides simultaneous speech, transcripts and hand-verified phonemes.

* [*CSTR NAM TIMIT*](https://homepages.inf.ed.ac.uk/jyamagis/page3/page57/page57.html): a parallel whispered speech dataset recorded simultaneously via a non-audible murmu (NAM) microphone, an omni-directional headset-mounted condenser microphone (a DPA 4035).

