#pragma once
#include <emmintrin.h>
#include <pmmintrin.h>	// MOVDDUP and ADDSUBPS are from SSE3 set
#include "complex.h"

// ==== Miscellaneous utilities ====
using XMVECTOR2 = __m128;
using XMVECTOR = __m128;

// Load 2D vector from memory
__forceinline XMVECTOR2 loadFloat2( const void* pointer )
{
	return _mm_castpd_ps( _mm_load_sd( (const double*)pointer ) );
}

// [ a, b, a, b ] where [ a, b ] are 2 floats at the address
__forceinline XMVECTOR loadFloat2Dup( const void* pointer )
{
	// Duplicating while loading is free, compare these two instructions:
	// https://uops.info/html-instr/MOVDDUP_XMM_M64.html
	// https://uops.info/html-instr/MOVSD_XMM_XMM_M64.html
	return _mm_castpd_ps( _mm_loaddup_pd( (const double*)pointer ) );
}

// Store a 2D vector
__forceinline void storeFloat2( void* pointer, XMVECTOR2 vec )
{
	_mm_store_sd( (double*)pointer, _mm_castps_pd( vec ) );
}

// [ a, b, c, d ] => [ c, d, a, b ]
__forceinline XMVECTOR flipHighLow( XMVECTOR v )
{
	return _mm_shuffle_ps( v, v, _MM_SHUFFLE( 1, 0, 3, 2 ) );
}

__forceinline XMVECTOR2 getHigh( XMVECTOR v )
{
	return _mm_movehl_ps( v, v );
}

// ==== Vectorizing fft_init ====

// pow( M_E, complex{ 0, valScalar } )
// https://en.wikipedia.org/wiki/Complex_number#Exponential_function, Euler's formula says the result is
// [ cos( val ), sin( val ) ]
XMVECTOR computeOmegaVec( const float valScalar );

// Same as above, for 2 different angles at once
XMVECTOR computeOmegaVec_x2( const XMVECTOR2 angles );

// Same as above, for 4 different angles at once
void __vectorcall computeOmegaVec_x4( const XMVECTOR angles, float* const destPointer );

#ifdef __AVX2__
#include <immintrin.h>
void __vectorcall computeOmegaVec_x8( const __m256 angles, float* const destPointer );
#endif

// ==== Vectorizing fft_run ====

// Multiply 2 complex numbers, the result vector is duplicated, [ a, b, a, b ] where a = real, b = imaginary
// The x input value must be duplicated the same way, for y the higher 2 lanes are ignored
__forceinline XMVECTOR multiplyComplex( XMVECTOR2 x, XMVECTOR2 y )
{
	// If the inputs are [ a, b ] and [ c, d ] the formula is [ ac - bd, ad + bc ]
	// BTW, it takes only 1 instruction less compared to scalar code, 5 versus 6.
	// The win comes from the fact that only 2 of them have >1 cycle latency, shuffles are really fast.

	// x is [ a, b, a, b ], duplicated while loading from memory
	y = _mm_unpacklo_ps( y, y );			// [ c, c, d, d ]
	const __m128 prod = _mm_mul_ps( x, y );	// [ ac, bc, ad, bd ]

	// Modern CPUs can run 2 of these per clock, just what we need as there's no data dependency
	const __m128 r1 = _mm_shuffle_ps( prod, prod, _MM_SHUFFLE( 2, 0, 2, 0 ) );	// [ ac, ad, ac, ad ]
	const __m128 r2 = _mm_shuffle_ps( prod, prod, _MM_SHUFFLE( 1, 3, 1, 3 ) );	// [ bd, bc, bd, bc ]

	// Pretty sure someone at Intel was thinking about complex numbers in early 2000s when they added addsubps to P4 Prescott
	const __m128 res = _mm_addsub_ps( r1, r2 );	// [ ac - bd, ad + bc, ac - bd, ad + bc ]
	return res;
}

// a1 = a1 + om * a2; a2 = a1 - om * a2
__forceinline void fftMainLoop( const complex* om, complex* a1c, complex* a2c )
{
	const XMVECTOR2 omega = loadFloat2Dup( om );
	const XMVECTOR2 a1 = loadFloat2( a2c );
	const XMVECTOR a0dup = loadFloat2Dup( a1c );

	XMVECTOR product = multiplyComplex( omega, a1 );
	product = _mm_xor_ps( product, _mm_setr_ps( 0, 0, -0.0f, -0.0f ) );
	const XMVECTOR result = _mm_add_ps( a0dup, product );
	storeFloat2( a1c, result );
	storeFloat2( a2c, getHigh( result ) );
}

// Shuffle mask for vrev64q_f32
constexpr int shuffleMask_rev64q = _MM_SHUFFLE( 2, 3, 0, 1 );

// Reverse elements in 64-bit doublewords (vector).
__forceinline XMVECTOR vrev64q_f32( XMVECTOR x )
{
	return _mm_shuffle_ps( x, x, shuffleMask_rev64q );
}

// Same as multiplyComplex, multiplies 2 numbers
__forceinline XMVECTOR multiplyComplex_x2( const XMVECTOR x, const XMVECTOR y )
{
	// If the inputs are [ a, b ] and [ c, d ] the formula is [ ac - bd, ad + bc ]

	const XMVECTOR x1 = _mm_moveldup_ps( x );	// [ a, a ]
	const XMVECTOR x2 = _mm_movehdup_ps( x );	// [ b, b ]
	const XMVECTOR yRev = vrev64q_f32( y );		// [ d, c ]

	const XMVECTOR prod1 = _mm_mul_ps( x1, y ); // [ ac, ad ]
	const XMVECTOR prod2 = _mm_mul_ps( x2, yRev ); // [ bd, bc ]

	return _mm_addsub_ps( prod1, prod2 );	// [ ac - bd, ad + bc ]
}

// Same as fftMainLoop, handles 2 complex numbers
__forceinline void fftMainLoop_x2( const complex* om, complex* a1c, complex* a2c )
{
	// Full-vector loads/stores are more efficient, especially so when they happen to be aligned.
	// Dual-channel DDR delivers exactly 128 bits per transaction.
	const XMVECTOR omega = _mm_loadu_ps( (const float*)om );
	const XMVECTOR a1 = _mm_loadu_ps( (const float*)a1c );
	const XMVECTOR a2 = _mm_loadu_ps( (const float*)a2c );

	const XMVECTOR product = multiplyComplex_x2( omega, a2 );

	_mm_storeu_ps( (float*)a1c, _mm_add_ps( a1, product ) );
	_mm_storeu_ps( (float*)a2c, _mm_sub_ps( a1, product ) );
}

#ifdef __AVX__
#include <immintrin.h>

// Same as multiplyComplex, multiplies 4 numbers
__forceinline __m256 multiplyComplex_x4( const __m256 x, const __m256 y )
{
	// If the inputs are [ a, b ] and [ c, d ] the formula is [ ac - bd, ad + bc ]

	const __m256 x1 = _mm256_moveldup_ps( x );	// [ a, a ]
	const __m256 x2 = _mm256_movehdup_ps( x );	// [ b, b ]
	const __m256 yRev = _mm256_permute_ps( y, shuffleMask_rev64q );		// [ d, c ]

	const __m256 prod1 = _mm256_mul_ps( x1, y ); // [ ac, ad ]
	const __m256 prod2 = _mm256_mul_ps( x2, yRev ); // [ bd, bc ]

	return _mm256_addsub_ps( prod1, prod2 );	// [ ac - bd, ad + bc ]
}

// Same as fftMainLoop, handles 4 complex numbers
__forceinline void fftMainLoop_x4( const complex* om, complex* a1c, complex* a2c )
{
	// Full-vector loads/stores are more efficient, especially so when they happen to be aligned.
	// Dual-channel DDR delivers exactly 128 bits per transaction.
	const __m256 omega = _mm256_loadu_ps( (const float*)om );
	const __m256 a1 = _mm256_loadu_ps( (const float*)a1c );
	const __m256 a2 = _mm256_loadu_ps( (const float*)a2c );

	const __m256 product = multiplyComplex_x4( omega, a2 );

	_mm256_storeu_ps( (float*)a1c, _mm256_add_ps( a1, product ) );
	_mm256_storeu_ps( (float*)a2c, _mm256_sub_ps( a1, product ) );
}
#else
__forceinline void fftMainLoop_x4( const complex* om, complex* a1c, complex* a2c )
{
	// When AVX is disabled in project settings, call fftMainLoop_x2 twice for the same effect.
	// The SSE version is only ~15% slower, BTW.
	fftMainLoop_x2( om, a1c, a2c );
	fftMainLoop_x2( om + 2, a1c + 2, a2c + 2 );
}
#endif