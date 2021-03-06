#pragma once
#include "fft.h"
#include "complex.h"

// ==== Vectorizing fft_init ====
// When PRECISE_TABLE is set in fft.h, these pieces of code are not used.

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

__forceinline __m128 negateHighHalf( __m128 v )
{
	return _mm_xor_ps( v, _mm_setr_ps( 0, 0, -0.0f, -0.0f ) );
}

// a1 = a1 + om * a2; a2 = a1 - om * a2
__forceinline void fftMainLoop( const complex* om, complex* a1c, complex* a2c )
{
	const XMVECTOR2 omegaDup = loadFloat2Dup( om );
	const XMVECTOR2 a1 = loadFloat2( a2c );
	const XMVECTOR a0dup = loadFloat2Dup( a1c );

	XMVECTOR product = multiplyComplex( omegaDup, a1 );
	// Negate the upper copy, to compute both ( a1 + product ) and ( a1 - product ) in one shot
	product = negateHighHalf( product );
	const XMVECTOR result = _mm_add_ps( a0dup, product );
	storeFloat2( a1c, result );
	storeFloat2( a2c, getHigh( result ) );
}

// Specialization of the above for omega = real 1.0
__forceinline void fftMainLoop_one( complex* a1c, complex* a2c )
{
	const XMVECTOR a1dup = loadFloat2Dup( a2c );
	const XMVECTOR a0dup = loadFloat2Dup( a1c );
	const XMVECTOR product = negateHighHalf( a1dup );
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
	const XMVECTOR x2 = _mm_movehdup_ps( x );	// [ b, b ]
	const XMVECTOR yRev = vrev64q_f32( y );		// [ d, c ]

	// const XMVECTOR prod1 = _mm_mul_ps( x1, y ); // [ ac, ad ]
	const XMVECTOR prod2 = _mm_mul_ps( x2, yRev ); // [ bd, bc ]

	const XMVECTOR x1 = _mm_moveldup_ps( x );	// [ a, a ]
	// return _mm_addsub_ps( prod1, prod2 );	// [ ac - bd, ad + bc ]
	return _mm_fmaddsub_ps( x1, y, prod2 );
}

// Same as fftMainLoop, handles 2 complex numbers
__forceinline void fftMainLoop_x2( const XMVECTOR omega, complex* a1c, complex* a2c )
{
	// Full-vector loads/stores are more efficient, especially so when they happen to be aligned.
	// Dual-channel DDR delivers exactly 128 bits per transaction.
	// Too bad VC++ 2017 optimizer is unaware, and fuses these loads into other instructions so the code does 2 loads from the same address :-(
	const XMVECTOR a1 = _mm_loadu_ps( (const float*)a1c );
	const XMVECTOR a2 = _mm_loadu_ps( (const float*)a2c );

	const XMVECTOR product = multiplyComplex_x2( omega, a2 );

	_mm_storeu_ps( (float*)a1c, _mm_add_ps( a1, product ) );
	_mm_storeu_ps( (float*)a2c, _mm_sub_ps( a1, product ) );
}

__forceinline void fftMainLoop_x2( const complex* om, complex* a1c, complex* a2c )
{
	const XMVECTOR omega = _mm_load_ps( (const float*)om );
	fftMainLoop_x2( omega, a1c, a2c );
}

// Specialization of the above for two omegas [ 1, 0 ], [ 0, -1 ]
__forceinline void fftMainLoop_span2( complex* a1c, complex* a2c )
{
	const XMVECTOR a1 = _mm_loadu_ps( (const float*)a1c );
	const XMVECTOR a2 = _mm_loadu_ps( (const float*)a2c );
	// The first omega is real 1.0, multiplication does nothing
	// The second number is [ 0, -1 ], multiplication by that number flips values then negates the imaginary one:
	// [ a, b ] * [ 0, -1 ] = [ b, -a ]
	XMVECTOR product = _mm_shuffle_ps( a2, a2, _MM_SHUFFLE( 2, 3, 1, 0 ) );
	product = _mm_xor_ps( product, _mm_setr_ps( 0, 0, 0, -0.0f ) );

	_mm_storeu_ps( (float*)a1c, _mm_add_ps( a1, product ) );
	_mm_storeu_ps( (float*)a2c, _mm_sub_ps( a1, product ) );
}

#ifdef __AVX__
#include <immintrin.h>

// Same as multiplyComplex, multiplies 4 numbers
__forceinline __m256 multiplyComplex_x4( const __m256 x, const __m256 y )
{
	// If the inputs are [ a, b ] and [ c, d ] the formula is [ ac - bd, ad + bc ]

	const __m256 x2 = _mm256_movehdup_ps( x );	// [ b, b ]
	const __m256 yRev = _mm256_permute_ps( y, shuffleMask_rev64q );		// [ d, c ]

	// const __m256 prod1 = _mm256_mul_ps( x1, y ); // [ ac, ad ]
	const __m256 prod2 = _mm256_mul_ps( x2, yRev ); // [ bd, bc ]

	// return _mm256_addsub_ps( prod1, prod2 );	// [ ac - bd, ad + bc ]
	const __m256 x1 = _mm256_moveldup_ps( x );	// [ a, a ]
	return _mm256_fmaddsub_ps( x1, y, prod2 );
}

// Same as fftMainLoop, handles 4 complex numbers
__forceinline void fftMainLoop_x4( const __m256 omega, complex* a1c, complex* a2c )
{
	const __m256 a1 = _mm256_loadu_ps( (const float*)a1c );
	const __m256 a2 = _mm256_loadu_ps( (const float*)a2c );

	const __m256 product = multiplyComplex_x4( omega, a2 );

	_mm256_storeu_ps( (float*)a1c, _mm256_add_ps( a1, product ) );
	_mm256_storeu_ps( (float*)a2c, _mm256_sub_ps( a1, product ) );
}

__forceinline void fftMainLoop_x4( const complex* om, complex* a1c, complex* a2c )
{
	const __m256 omega = _mm256_load_ps( (const float*)om );
	fftMainLoop_x4( omega, a1c, a2c );
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