#include "stdafx.h"
#include <xmmintrin.h>
#include <immintrin.h>
#include <complex>
#include "macro.h"
#include "fft.h"
#include "fftSimd.h"
#include "reverseBits.h"

static bool initialized = false;
// The original code had this array transposed, i.e. [ K ][ MAX_SAMPLES ]
// Despite the inner loops of both fft_init and fft_run has a loop over K.
// Transposing to the correct layout doubled the performance right away.
static complex omega_vec[ MAX_SAMPLES ][ K ];

// Old implementation no longer in use, see reverseBits.h and .cpp
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

// Processors have an instruction for that, very fast.
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
			omega_vec[ n ][ k ] = complex{ result.real(), result.imag() };
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
			storeComplex( omega_vec[ n ][ k ], result );
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
			_mm_storeu_ps( (float*)( &omega_vec[ n ][ k ] ), result );
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

			float* dest = (float*)( &omega_vec[ n ][ k ] );
			_mm_storeu_ps( dest, result.first );
			_mm_storeu_ps( dest + 4, result.second );
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

// Fill continuous region of complex numbers with zeros.
__forceinline void writeZeros( complex* pointer, uint32_t count )
{
	static_assert( sizeof( complex ) == 8 );
	complex* const pointerEndAligned = pointer + ( count & ~1u );
	const __m128 zero = _mm_setzero_ps();
	for( ; pointer < pointerEndAligned; pointer += 2 )
		_mm_storeu_ps( (float*)pointer, zero );
	if( 0 != ( count % 2 ) )
		storeFloat2( pointer, zero );
}

void fft_run( const float *input_data, complex *output_data, uint32_t N, uint32_t channels )
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
#if 0
			for( unsigned int i = N; i < new_N; i++ )
				output_data[ i ] = 0.0f;
#else
			writeZeros( &output_data[ N ], new_N - N );
#endif
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
					fftMainLoop( &omega_vec[ n ][ k ], &output_data[ k + j ], &output_data[ k + j + wingspan ] );
				}
			}
			wingspan *= 2;
		}
	}
}