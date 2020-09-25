#include "stdafx.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "macro.h"
#include "fft.h"

static bool initialized = false;
static std::complex<float> omega_vec[ K ][ MAX_SAMPLES ];

static inline unsigned int reverse_bits( unsigned int n, unsigned int num_bits ) {
	int i, j;
	register unsigned int res = 0;

	i = 0;
	j = num_bits;
	while( i <= j ) {
		unsigned int lower_mask = 1 << i;
		unsigned int upper_mask = ( 1 << num_bits ) >> i;
		unsigned int shift = j - i;
		res |= ( ( n >> shift ) & lower_mask ) | ( ( n << shift ) & upper_mask );
		i++;
		j--;
	}
	return res;
}

static inline unsigned int get_msb( unsigned int v ) {
	static const unsigned int b[] = { 0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000 };
	static const unsigned int S[] = { 1, 2, 4, 8, 16 };

	register unsigned int r = 0;

	if( v & b[ 4 ] )
	{
		v >>= S[ 4 ];
		r |= S[ 4 ];
	}
	if( v & b[ 3 ] )
	{
		v >>= S[ 3 ];
		r |= S[ 3 ];
	}
	if( v & b[ 2 ] )
	{
		v >>= S[ 2 ];
		r |= S[ 2 ];
	}
	if( v & b[ 1 ] )
	{
		v >>= S[ 1 ];
		r |= S[ 1 ];
	}
	if( v & b[ 0 ] )
	{
		v >>= S[ 0 ];
		r |= S[ 0 ];
	}
	return r;
}

void fft_init()
{
	// constexpr complex I{ 0, 1 };
	constexpr complex E{ (float)M_E, 0 };
	constexpr float mul = (float)( -2 * M_PI );
#pragma omp parallel for
	for( unsigned int n = 0; n < MAX_SAMPLES; n++ )
	{
		const float mulDivN = mul / n;
		for( unsigned int k = 0; k < K; k++ )
		{
			const float imag = (float)k * mulDivN;
			std::complex<float> exp{ 0, imag };
			omega_vec[ k ][ n ] = std::pow( E, exp );
		}
	}
	initialized = true;
}

void fft_run(
	const float *input_data,
	complex *output_data,
	unsigned int N,
	unsigned int channels )
{
	assert( initialized );

	{
		unsigned int msb;

		for( unsigned int i = 0, j = 0; i < N; j++, i += channels )
			/* Taking just the left channel for now... */
			output_data[ j ] = input_data[ i ];

		N = N / channels;
		assert( N <= MAX_SAMPLES );
		msb = get_msb( N );

		if( ( N & ( N - 1 ) ) )
		{
			/* Pad out so FFT is a power of 2. */
			msb = msb + 1;
			unsigned int new_N = 1 << msb;
			for( unsigned int i = N; i < new_N; i++ )
				output_data[ i ] = 0.0f;

			N = new_N;
		}

		/* Reverse the input array. */
		unsigned int hi_bit = msb - 1;
		for( unsigned int i = 0; i < N; i++ ) 
		{
			unsigned int r = reverse_bits( i, hi_bit );
			if( i < r )
				std::swap( output_data[ i ], output_data[ r ] );
		}
	}

	{
		/* Simple radix-2 DIT FFT */
		unsigned int wingspan = 1;
		while( wingspan < N ) 
		{
			unsigned int n = wingspan * 2;
			for( unsigned int j = 0; j < N; j += wingspan * 2 ) 
			{
				for( unsigned int k = 0; k < wingspan; k++ )
				{
					complex omega = omega_vec[ k ][ n ];
					complex a0 = output_data[ k + j ];
					complex a1 = output_data[ k + j + wingspan ];
					output_data[ k + j ] = a0 + omega * a1;
					output_data[ k + j + wingspan ] = a0 - omega * a1;
				}
			}
			wingspan *= 2;
		}
	}
}