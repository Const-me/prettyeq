#pragma once
#include <stdint.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <smmintrin.h>

#define MAX_SAMPLES_LOG_2 12
#define MAX_SAMPLES ( 1 << MAX_SAMPLES_LOG_2 )
#define K (MAX_SAMPLES / 2)

#define FFT_BUCKET_WIDTH(NUM_SAMPLES) (44100/(NUM_SAMPLES))
#define FFT_SAMPLE_TO_FREQ(NUM_SAMPLES, SAMPLE_INDEX) (44100*(SAMPLE_INDEX)/(NUM_SAMPLES))
#define FFT_FREQ_TO_SAMPLE(NUM_SAMPLES, FREQ) ((int)roundf((FREQ)*(NUM_SAMPLES)/44100))

// Use double-precision math for the table.
// Much slower than vectorized version, but still pretty fast overall, well under 0.5ms on my PC.
#define FFT_PRECISE_TABLE true

// Use FMA3 instructions. Only changes something when building without AVX.
#define FFT_USE_FMA3 true

#include "simdUtils.hpp"
#include "complex.h"

inline float FFT_PSD( const complex& c )
{
#if STD_COMPLEX
	return std::abs( c );
#else
	__m128 v = loadFloat2( &c );
	// Funny enough, _mm_dp_ps appears to be slightly slower here, despite 1 instruction versus 3.
	v = _mm_mul_ps( v, v );
	v = _mm_add_ss( v, _mm_movehdup_ps( v ) );
	v = _mm_sqrt_ss( v );
	return _mm_cvtss_f32( v );
#endif
}

void fft_init();

// The output buffer size must be N rounded up to the next power of 2
void fft_run(
	const float *input_data,
	complex *output_data,
	uint32_t N,
	uint32_t channels );