#include "stdafx.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "macro.h"
#include "fft.h"
#include <xmmintrin.h>
#include <immintrin.h>
#include "fftSimd.h"
#include "complex.h"
#include "reverseBits.h"
#include <complex>

static bool initialized = false;
static complex omega_vec[ K ][ MAX_SAMPLES ];

static inline unsigned int reverse_bits( unsigned int n, unsigned int num_bits )
{
	int i, j;
	unsigned int res = 0;

	i = 0;
	j = num_bits;
	while( i <= j )
	{
		unsigned int lower_mask = 1 << i;
		unsigned int upper_mask = ( 1 << num_bits ) >> i;
		unsigned int shift = j - i;
		res |= ( ( n >> shift ) & lower_mask ) | ( ( n << shift ) & upper_mask );
		i++;
		j--;
	}
	return res;
}

/* static inline unsigned int get_msb( unsigned int v )
{
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
} */
static inline unsigned int get_msb( unsigned int v )
{
	assert( 0 != v );
#ifdef _MSC_VER
	DWORD index;
	_BitScanReverse( &index, v );
	return index;
#else
	return 31 - __builtin_clz( v );
#endif
}

/* inline complex computeOmega( unsigned int k, unsigned int n )
{
	constexpr float XM_E = (float)M_E;
	constexpr float mul = (float)( -2 * M_PI );
	const float imag = mul * (float)k / (float)n;

	float s, c;
	XMScalarSinCos( &s, &c, imag );
	return complex{ c, s };
} */

__forceinline void storeComplex( complex& dest, __m128 val )
{
	storeFloat2( &dest, val );
}

static void fft_init_std()
{
	constexpr float XM_E = (float)M_E;
	constexpr float mul = (float)( -2 * M_PI );

	for( unsigned int n = 0; n < MAX_SAMPLES; n++ )
	{
		const float mulDivN = ( n != 0 ) ? mul / n : 0.0f;
		for( unsigned int k = 0; k < K; k++ )
		{
			const float imag = (float)k * mulDivN;
			const std::complex<float> pp{ 0, imag };
			const std::complex<float> result = std::pow( XM_E, pp );
			omega_vec[ k ][ n ] = complex{ result.real(), result.imag() };
		}
	}
}

static void fft_init_simd()
{
	constexpr float mul = (float)( -2 * M_PI );
	for( unsigned int n = 0; n < MAX_SAMPLES; n++ )
	{
		const float mulDivN = ( n != 0 ) ? mul / n : 0.0f;
		for( unsigned int k = 0; k < K; k++ )
		{
			const float imag = (float)k * mulDivN;
			const __m128 result = computeOmegaVec( imag );
			storeComplex( omega_vec[ k ][ n ], result );
		}
	}
}

static void fft_init_x2()
{
	constexpr float mul = (float)( -2 * M_PI );

	const __m128i kVecIncrement = _mm_set1_epi32( 2 );
	const __m128i kVecInitial = _mm_setr_epi32( 0, 1, 2, 3 );
	static_assert( 0 == ( K % 2 ) );
	for( unsigned int n = 0; n < MAX_SAMPLES; n++ )
	{
		const float mulDivN_Scalar = ( n != 0 ) ? mul / n : 0.0f;
		const __m128 mulDivN = _mm_set1_ps( mulDivN_Scalar );
		__m128i kVec = kVecInitial;
		for( unsigned int k = 0; k < K; k += 2, kVec = _mm_add_epi32( kVec, kVecIncrement ) )
		{
			const __m128 imag = _mm_mul_ps( _mm_cvtepi32_ps( kVec ), mulDivN );
			const __m128 result = computeOmegaVec_x2( imag );

			storeComplex( omega_vec[ k ][ n ], result );
			storeComplex( omega_vec[ k + 1 ][ n ], getHigh( result ) );
		}
	}
}

static void fft_init_x4()
{
	constexpr float mul = (float)( -2 * M_PI );

	const __m128i kVecIncrement = _mm_set1_epi32( 4 );
	const __m128i kVecInitial = _mm_setr_epi32( 0, 1, 2, 3 );
	static_assert( 0 == ( K % 4 ) );
	for( unsigned int n = 0; n < MAX_SAMPLES; n++ )
	{
		const float mulDivN_Scalar = ( n != 0 ) ? mul / n : 0.0f;
		const __m128 mulDivN = _mm_set1_ps( mulDivN_Scalar );
		__m128i kVec = kVecInitial;
		for( unsigned int k = 0; k < K; k += 4, kVec = _mm_add_epi32( kVec, kVecIncrement ) )
		{
			const __m128 imag = _mm_mul_ps( _mm_cvtepi32_ps( kVec ), mulDivN );
			const std::pair<__m128, __m128> result = computeOmegaVec_x4( imag );

			storeComplex( omega_vec[ k ][ n ], result.first );
			storeComplex( omega_vec[ k + 1 ][ n ], getHigh( result.first ) );
			storeComplex( omega_vec[ k + 2 ][ n ], result.second );
			storeComplex( omega_vec[ k + 3 ][ n ], getHigh( result.second ) );
		}
	}
}

void fft_init()
{
	// fft_init_std();
	// fft_init_simd();
	fft_init_x4();
	// fft_init_x2();
	initialized = true;
}

void fft_run( const float *input_data, complex *output_data, unsigned int N, unsigned int channels )
{
	assert( initialized );

	{
		for( unsigned int i = 0, j = 0; i < N; j++, i += channels )
			/* Taking just the left channel for now... */
			output_data[ j ] = input_data[ i ];

		N = N / channels;
		assert( N <= MAX_SAMPLES );
		unsigned int msb = get_msb( N );

		if( ( N & ( N - 1 ) ) )
		{
			/* Pad out so FFT is a power of 2. */
			msb++;
			unsigned int new_N = 1 << msb;
			for( unsigned int i = N; i < new_N; i++ )
				output_data[ i ] = 0.0f;

			N = new_N;
		}

		/* Reverse the input array. */
		unsigned int hi_bit = msb - 1;
		const ReverseBits reverseBits{ msb };
		for( unsigned int i = 0; i < N; i++ )
		{
#if 0
			unsigned int r1 = reverse_bits( i, hi_bit );
			unsigned int r2 = reverseBits( i );
			assert( r1 == r2 );
			const unsigned int r = r1;
#else
			const unsigned int r = reverseBits( i );
#endif
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
					/* complex omega = omega_vec[ k ][ n ];
					// complex omega = computeOmega( k, n );
					complex a0 = output_data[ k + j ];
					complex a1 = output_data[ k + j + wingspan ];
					// output_data[ k + j ] = a0 + omega * a1;
					// output_data[ k + j + wingspan ] = a0 - omega * a1;
					const complex prod = omega * a1;
					output_data[ k + j ] = a0 + prod;
					output_data[ k + j + wingspan ] = a0 - prod; */
					fftMainLoop( omega_vec[ k ][ n ], output_data[ k + j ], output_data[ k + j + wingspan ] );
				}
			}
			wingspan *= 2;
		}
	}
}