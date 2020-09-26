#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include "complex.h"
#include "fftSimd.h"
#include <smmintrin.h>

#define MAX_SAMPLES 4096
#define K (MAX_SAMPLES / 2)

#define FFT_BUCKET_WIDTH(NUM_SAMPLES) (44100/(NUM_SAMPLES))
#define FFT_SAMPLE_TO_FREQ(NUM_SAMPLES, SAMPLE_INDEX) (44100*(SAMPLE_INDEX)/(NUM_SAMPLES))
#define FFT_FREQ_TO_SAMPLE(NUM_SAMPLES, FREQ) ((int)roundf((FREQ)*(NUM_SAMPLES)/44100))

inline float FFT_PSD( complex c )
{
#if STD_COMPLEX
	return std::abs( c );
#else
	__m128 v = loadFloat2( &c );
	v = _mm_dp_ps( v, v, 0b110001 );
	v = _mm_sqrt_ss( v );
	return _mm_cvtss_f32( v );
#endif
}

void fft_init();
void fft_run(
	const float *input_data,
	complex *output_data,
	unsigned int N,
	unsigned int channels );