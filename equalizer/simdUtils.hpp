#pragma once
// ==== Miscellaneous utilities ====
#include <emmintrin.h>
#include <pmmintrin.h>	// MOVDDUP and ADDSUBPS are from SSE3 set

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

// ==== FMA ====
#if FFT_USE_FMA3
#include <immintrin.h>	// FMA
#else
// Workaround if you don't want to require FMA3 instruction set:
// https://en.wikipedia.org/wiki/FMA_instruction_set#CPUs_with_FMA3

__forceinline __m128 fmadd_ps( __m128 a, __m128 b, __m128 c )
{
	return _mm_add_ps( _mm_mul_ps( a, b ), c );
}
__forceinline __m128 fnmadd_ps( __m128 a, __m128 b, __m128 c )
{
	return _mm_sub_ps( c, _mm_mul_ps( a, b ) );
}
__forceinline __m128 fnmadd_ss( __m128 a, __m128 b, __m128 c )
{
	return _mm_sub_ss( c, _mm_mul_ss( a, b ) );
}
__forceinline __m128 fmaddsub_ps( __m128 a, __m128 b, __m128 c )
{
	return _mm_addsub_ps( _mm_mul_ps( a, b ), c );
}
#define _mm_fmadd_ps fmadd_ps
#define _mm_fnmadd_ps fnmadd_ps
#define _mm_fnmadd_ss fnmadd_ss
#define _mm_fmaddsub_ps fmaddsub_ps
#endif