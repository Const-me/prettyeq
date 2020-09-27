This project is an example how to manually vectorize stuff.The algorithm is from there: https://github.com/keur/prettyeq/tree/master/equalizerThe computational parts weren’t too complex, that’s why I thought it shouldn’t take too long to find out how much faster I can make it.I can’t test the results acoustically because I use Windows, but the OP has implemented a few tests, they pass with my version.## Performance Improvements	Baseline:	[FFT Init 4096 samples] time: 377.060900 ms	[FFT Run 512 samples] time: 15.700000 us	Manually vectorized AVX2:	[FFT Init 4096 samples] time: 23.959500 ms	[FFT Run 512 samples] time: 4.300000 us	Manually vectorized SSE:	[FFT Init 4096 samples] time: 28.345400 ms	[FFT Run 512 samples] time: 5.000000 usManual vectorization improved performance of `fft_init` by a factor of 13-15, and `fft_run` by 3-4.Not bad, eh?If you want to reproduce the results, you'll need Windows and Visual Studio, open `equalizer\FftTest.sln` solution.I was using Visual Studio 2017, the freeware community edition.## DisclaimersThe only thing that builds is the FFT test command-line application.Because I have no way to test that acoustically, could be numerical errors despite the passing test. However, conceptually the algorithm is relatively simple, should be fixable.I’m pretty sure the OP’s code is slightly faster than my baseline version. Two reasons. The OP did it in C, I ported to C++ and replaced with `std::complex<float>`, probably has some overhead. Also, in some cases gcc and clang can emit slightly faster code (in other cases it's the opposite, though).The code is research-quality, if you gonna reuse in production software you might need to clean up these `#if 0` and other temporary shenanigans.I have only tested 64-bit builds. The non-AVX version requires SSE3 , SSE 4.1 and FMA3. Fortunately, you gonna try very hard to find a working computer without the first two, they're supported in all CPUs made after 2008 or so, computers usually don't live that long. FMA3 is newer unfortunately, see the preprocessor block at the start of `fftSimd.h` source file for the workaround. I haven’t tested the performance but it shouldn’t be too bad.If you gonna port that from VC++ to gcc or clang, among other things you’ll need to replace `__AVX__` and  `__AVX2__` feature test macros with something else, these two probably define another one.Original readme goes below.----------------## PrettyEQprettyeq is a system-wide paramateric equalizer for pulseaudio. This softwareis in alpha. Use at your own discretion.![prettyeq demo](https://i.fluffy.cc/0GFcjGmbrtCgnbRSjd4xjDcf7h6qNk4Q.gif)### UsageWhen the program is executed all pulseaudio streams will be routed through theequalizer. With no filters activated prettyeq acts as a passthrough sink.Right now prettyeq only supports two-channel sound.##### Filter typesprettyeq has three filter types:* one **low shelf** filter mounted at 20Hz* one **high shelf** filter mounted at 20kHz* five **peakingEQ** filters that can move freely##### ControlsClick and drag points to boost and cut frequency bands. The dB gain range is±12dB. Filter bandwidth and slope can be changed with the mousewheel.##### QuittingIf your desktop has a system tray, the close button will hide the GUI but theequalizer will still be in effect. There are context menus in the applicationand tray that have a "exit" option to quit the application.