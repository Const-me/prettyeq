#pragma once
#include <emmintrin.h>
#include <pmmintrin.h>	// MOVDDUP is from SSE3 set
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

// Same as above, for 4 different angles at once
std::pair<XMVECTOR, XMVECTOR> computeOmegaVec_x4( const XMVECTOR angles );

// ==== Vectorizing fft_run ====

// Multiply 2 complex numbers, the result vector is duplicated [ a, b, a, b ] where a = real, b = imaginary
__forceinline XMVECTOR multiplyComplex( XMVECTOR2 x, XMVECTOR2 y )
{
	// If the inputs are [ a, b ] and [ c, d ] the formula is [ ac - bd, ad + bc ]
	// BTW, it takes same count of instructions, 6, as scalar code.
	// The win comes from the fact that only 2 of them have >1 cycle latency: shuffles and xor are _really_ fast.

	x = _mm_unpacklo_ps( x, x ); // [ a, a, b, b ]
	y = _mm_shuffle_ps( y, y, _MM_SHUFFLE( 0, 1, 1, 0 ) );	// [ c, d, d, c ]
	__m128 prod = _mm_mul_ps( x, y );	// [ ac, ad, bd, bc ]
	prod = _mm_xor_ps( prod, _mm_setr_ps( 0, 0, -0.0f, 0 ) );	// [ ac, ad, -bd, bc ]
	return _mm_add_ps( prod, flipHighLow( prod ) );
}

using complex = std::complex<float>;

__forceinline void fftMainLoop( const complex& om, complex& acc1, complex& acc2 )
{
	const XMVECTOR2 omega = loadFloat2( &om );
	// We want a0 + omega * a1 and a0 - omega * a1
	// Duplicating while loading is free, compare these two instructions:
	// https://uops.info/html-instr/MOVDDUP_XMM_M64.html
	// https://uops.info/html-instr/MOVSD_XMM_XMM_M64.html
	const XMVECTOR a0 = loadFloat2Dup( &acc1 );
	const XMVECTOR2 a1 = loadFloat2( &acc2 );

	XMVECTOR product = multiplyComplex( omega, a1 );
	product = _mm_xor_ps( product, _mm_setr_ps( 0, 0, -0.0f, -0.0f ) );
	const XMVECTOR result = _mm_add_ps( a0, product );
	storeFloat2( &acc1, result );
	storeFloat2( &acc2, getHigh( result ) );
}