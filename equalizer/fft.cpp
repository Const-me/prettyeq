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
// static complex omega_vec[ MAX_SAMPLES ][ K ];

// Actually, we only use rows of that table with row indices being powers of 2.
// No need to waste memory and CPU time computing the rest of these rows.
alignas( 32 ) static complex omega_vec_log[ MAX_SAMPLES_LOG_2 ][ K ];

// Index of the most significant non-zero bit. Processors have an instruction for that, very fast.
static inline uint32_t get_msb( uint32_t v )
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

constexpr float minus2pi = (float)( -2.0 * M_PI );

// Initialize lookup table using std::complex<float> from the standard library
static void fft_init_std()
{
	constexpr float XM_E = (float)M_E;
	for( uint32_t nl = 0; nl < MAX_SAMPLES_LOG_2; nl++ )
	{
		const uint32_t n = 1u << nl;
		const float mulDivN = minus2pi / n;
		for( unsigned int k = 0; k < K; k++ )
		{
			const float imag = (float)k * mulDivN;
			const std::complex<float> pp{ 0, imag };
			const std::complex<float> result = std::pow( XM_E, pp );
			omega_vec_log[ nl ][ k ] = complex{ result.real(), result.imag() };
		}
	}
}

// Initialize the table with SSE SIMD
static void fft_init_x4()
{
	const __m128i kVecIncrement = _mm_set1_epi32( 4 );
	const __m128i kVecInitial = _mm_setr_epi32( 0, 1, 2, 3 );
	static_assert( 0 == ( K % 4 ) );
	for( uint32_t nl = 0; nl < MAX_SAMPLES_LOG_2; nl++ )
	{
		const uint32_t n = 1u << nl;
		const float mulDivN_Scalar = minus2pi / n;
		const __m128 mulDivN = _mm_set1_ps( mulDivN_Scalar );
		__m128i kVec = kVecInitial;
		for( unsigned int k = 0; k < K; k += 4, kVec = _mm_add_epi32( kVec, kVecIncrement ) )
		{
			const __m128 imag = _mm_mul_ps( _mm_cvtepi32_ps( kVec ), mulDivN );
			float* const dest = (float*)( &omega_vec_log[ nl ][ k ] );
			computeOmegaVec_x4( imag, dest );
		}
	}
}
#ifdef __AVX2__
// Initialize the table with AVX2 SIMD
static void fft_init_x8()
{
	const __m256i kVecIncrement = _mm256_set1_epi32( 8 );
	const __m256i kVecInitial = _mm256_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7 );
	static_assert( 0 == ( K % 8 ) );
	for( uint32_t nl = 0; nl < MAX_SAMPLES_LOG_2; nl++ )
	{
		const uint32_t n = 1u << nl;
		const float mulDivN_Scalar = minus2pi / n;
		const __m256 mulDivN = _mm256_set1_ps( mulDivN_Scalar );
		__m256i kVec = kVecInitial;
		for( unsigned int k = 0; k < K; k += 8, kVec = _mm256_add_epi32( kVec, kVecIncrement ) )
		{
			const __m256 imag = _mm256_mul_ps( _mm256_cvtepi32_ps( kVec ), mulDivN );
			float* const dest = (float*)( &omega_vec_log[ nl ][ k ] );
			computeOmegaVec_x8( imag, dest );
		}
	}
}
#endif

void fft_init()
{
#ifdef __AVX2__
	fft_init_x8();
#else
	// fft_init_std();
	fft_init_x4();
#endif
	initialized = true;
}

// Fill continuous span of complex numbers with zeros.
static __forceinline void writeZeros( complex* pointer, uint32_t count )
{
	static_assert( sizeof( complex ) == 8 );
	complex* const pointerEndAligned = pointer + ( count & ( ~1u ) );
	const __m128 zero = _mm_setzero_ps();
	for( ; pointer < pointerEndAligned; pointer += 2 )
		_mm_storeu_ps( (float*)pointer, zero );
	if( 0 != ( count % 2 ) )
		storeFloat2( pointer, zero );
}

// Template version for small values of wingspan, unrolls the inner loop for better performance
template<uint32_t wingspan>
static __forceinline void fft_run_main_unroll( uint32_t N, complex *output_data )
{
	for( uint32_t j = 0; j < N; j += wingspan * 2 )
	{
		complex* out1 = &output_data[ j ];
		complex* out2 = &output_data[ j + wingspan ];
		if constexpr( 1 == wingspan )
			fftMainLoop_one( out1, out2 );
		else if constexpr( 2 == wingspan )
			fftMainLoop_span2( out1, out2 );
		else
			assert( false );
	}
}

template<>
static __forceinline void fft_run_main_unroll<4>( uint32_t N, complex *output_data )
{
	constexpr uint32_t wingspan = 4;
	constexpr uint32_t n = wingspan * 2;	// 8 = 2^3
	// Loading these things into registers outside of the inner loop is what gives the performance win
	const float* omega_src = (const float*)&omega_vec_log[ 3 ][ 0 ];
#ifdef __AVX__
	// Note we don't need AVX2 for this part, it only does float math and AVX1 is enough.
	const __m256 omega = _mm256_load_ps( omega_src );
#else
	const __m128 omegaLow = _mm_load_ps( omega_src );
	const __m128 omegaHigh = _mm_load_ps( omega_src + 4 );
#endif

	for( uint32_t j = 0; j < N; j += wingspan * 2 )
	{
		complex* out1 = &output_data[ j ];
		complex* out2 = &output_data[ j + wingspan ];
#ifdef __AVX__
		fftMainLoop_x4( omega, out1, out2 );
#else
		fftMainLoop_x2( omegaLow, out1, out2 );
		fftMainLoop_x2( omegaHigh, out1 + 2, out2 + 2 );
#endif
	}
}

template<>
static __forceinline void fft_run_main_unroll<8>( uint32_t N, complex *output_data )
{
	constexpr uint32_t wingspan = 8;
	constexpr uint32_t n = wingspan * 2;	// 16 = 2^4
	const float* omega_src = (const float*)&omega_vec_log[ 4 ][ 0 ];
#ifdef __AVX__
	const __m256 omega0 = _mm256_load_ps( omega_src );
	const __m256 omega1 = _mm256_load_ps( omega_src + 8 );
#else
	const __m128 omega0 = _mm_load_ps( omega_src );
	const __m128 omega1 = _mm_load_ps( omega_src + 4 );
	const __m128 omega2 = _mm_load_ps( omega_src + 8 );
	const __m128 omega3 = _mm_load_ps( omega_src + 12 );
#endif

	for( uint32_t j = 0; j < N; j += wingspan * 2 )
	{
		complex* out1 = &output_data[ j ];
		complex* out2 = &output_data[ j + wingspan ];
#ifdef __AVX__
		fftMainLoop_x4( omega0, out1, out2 );
		fftMainLoop_x4( omega1, out1 + 4, out2 + 4 );
#else
		fftMainLoop_x2( omega0, out1, out2 );
		fftMainLoop_x2( omega1, out1 + 2, out2 + 2 );
		fftMainLoop_x2( omega2, out1 + 4, out2 + 4 );
		fftMainLoop_x2( omega3, out1 + 6, out2 + 6 );
#endif
	}
}

// Non-template version when we actually want a loop. The inner loop body handles 8 complex numbers per iteration.
static __forceinline void fft_run_main_n( uint32_t wingspan, uint32_t N, complex *output_data )
{
	assert( wingspan >= 8 && 0 == ( wingspan % 8 ) );
	const uint32_t n = wingspan * 2;
	const complex* const omegaBegin = &omega_vec_log[ get_msb( n ) ][ 0 ];
	const complex* const omegaEnd = omegaBegin + wingspan;

	for( uint32_t j = 0; j < N; j += wingspan * 2 )
	{
		complex* out1 = &output_data[ j ];
		complex* out2 = &output_data[ j + wingspan ];
		for( const complex* om = omegaBegin; om < omegaEnd; om += 8, out1 += 8, out2 += 8 )
		{
			fftMainLoop_x4( om, out1, out2 );
			fftMainLoop_x4( om + 4, out1 + 4, out2 + 4 );
		}
	}
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
			writeZeros( &output_data[ N ], new_N - N );
			N = new_N;
		}

		/* Reverse the input array. */
		unsigned int hi_bit = msb - 1;
		const ReverseBits reverseBits{ msb };
		for( unsigned int i = 0; i < N; i++ )
		{
			const unsigned int r = reverseBits( i );
			if( i < r )
				std::swap( output_data[ i ], output_data[ r ] );
		}
	}

	{
		// wingspan 1
		if( 1 >= N ) return;
		fft_run_main_unroll<1>( N, output_data );
		// wingspan 2
		if( 2 >= N ) return;
		fft_run_main_unroll<2>( N, output_data );
		// wingspan 4
		if( 4 >= N ) return;
		fft_run_main_unroll<4>( N, output_data );
		// wingspan 8
		if( 8 >= N ) return;
		fft_run_main_unroll<8>( N, output_data );

		// For 16 and more we use actual inner loop; small loops are bad for branch predictor, the exit condition changes too often.
		for( uint32_t wingspan = 16; wingspan < N; wingspan *= 2 )
			fft_run_main_n( wingspan, N, output_data );
	}
}