#include "stdafx.h"
#include "reverseBits.h"
#include <tmmintrin.h>
#include <smmintrin.h>
#include <assert.h>

__forceinline uint32_t lookup4( uint32_t x, __m128i table )
{
	__m128i vec = _mm_cvtsi32_si128( (int)x );
	vec = _mm_shuffle_epi8( table, vec );
	return (uint32_t)_mm_cvtsi128_si32( vec );
}

template<int n>
static uint32_t __vectorcall reverseBitsImpl( uint32_t x, __m128i table )
{
	static_assert( n >= 0 && n <= 32 );
	if constexpr( n <= 1 )
		return x;

	if constexpr( n <= 4 )
		x = lookup4( x, table );
	else if constexpr( n <= 8 )
	{
		x = ( ( x & 0xF0 ) << 4 ) | ( x & 0xF );
		x = lookup4( x, table );
		x = ( ( x & 0x0F00 ) >> 8 ) | ( ( x & 0x0F ) << 4 );
	}
	else if constexpr( n <= 12 )
	{
		x = ( ( x & 0xF00 ) << 8 ) | ( ( x & 0xF0 ) << 4 ) | ( x & 0xF );
		x = lookup4( x, table );
		x = ( ( x & 0x0F0000 ) >> 16 ) | ( ( x & 0x0F00 ) >> 4 ) | ( ( x & 0x0F ) << 8 );
	}
	else
	{
		// Shifting more of them the same way would take too many scalar instructions.
		// Vectorizing that part as well, _mm_shuffle_epi8, bitwise and shift instructions all have 1 cycle latency on modern CPUs.

		// Reverse every byte of the value
		__m128i vec = _mm_cvtsi32_si128( (int)x );
		// Reverse every sequence of 4 bits
		__m128i low = _mm_and_si128( vec, _mm_set1_epi8( 0x0F ) );
		__m128i high = _mm_and_si128( vec, _mm_set1_epi8( (char)0xF0 ) );
		high = _mm_srli_epi32( high, 4 );
		low = _mm_shuffle_epi8( table, low );
		high = _mm_shuffle_epi8( table, high );
		// Marge nibbles back into bytes. Note we shifting low nibbles to the higher positions.
		low = _mm_slli_epi32( low, 4 );
		const __m128i flippedBytes = _mm_or_si128( high, low );

		// Reverse bytes, either 2, 3 or 4 of them, with another _mm_shuffle_epi8
		uint32_t shufBytesScalar;
		if constexpr( n <= 16 )
			shufBytesScalar = 0xFFFF0001;
		else if constexpr( n <= 24 )
			shufBytesScalar = 0xFF000102;
		else
			shufBytesScalar = 0x00010203;

		const __m128i allSet = _mm_cmpeq_epi32( flippedBytes, flippedBytes );
		const __m128i shuffleMask = _mm_blend_epi16( allSet, _mm_cvtsi32_si128( shufBytesScalar ), 0b11 );
		const __m128i resultVec = _mm_shuffle_epi8( flippedBytes, shuffleMask );
		// Move the integer to scalar portion of the CPU
		x = (uint32_t)_mm_cvtsi128_si32( resultVec );

		// Final shift of the entire integer into position
		constexpr int shiftAmount8 = ( 8 - ( n % 8 ) ) % 8;
		if constexpr( 0 == shiftAmount8 )
			return x;
		return x >> shiftAmount8;
	}

	constexpr int shiftAmount4 = ( 4 - ( n % 4 ) ) % 4;
	if constexpr( 0 == shiftAmount4 )
		return x;
	return x >> shiftAmount4;
}

static const std::array<ReverseBits::pfn, 33> s_functions =
{
#define DF4( i ) &reverseBitsImpl<i>, &reverseBitsImpl<i+1>, &reverseBitsImpl<i+2>, &reverseBitsImpl<i+3>
	DF4( 0 ),
	DF4( 4 ),
	DF4( 8 ),
	DF4( 12 ),
	DF4( 16 ),
	DF4( 20 ),
	DF4( 24 ),
	DF4( 28 ),
	&reverseBitsImpl<32>
};

// Make a lookup table for flipping bits in 4-bit segments
constexpr char lookupEntry( uint8_t val )
{
	return (char)( ( ( val & 1 ) << 3 ) | ( ( val & 2 ) << 1 ) | ( ( val & 4 ) >> 1 ) | ( ( val & 8 ) >> 3 ) );
}

#define RB lookupEntry
static const __m128i s_reverseBitsLookup = _mm_setr_epi8(
	RB( 0 ), RB( 1 ), RB( 2 ), RB( 3 ),
	RB( 4 ), RB( 5 ), RB( 6 ), RB( 7 ),
	RB( 8 ), RB( 9 ), RB( 10 ), RB( 11 ),
	RB( 12 ), RB( 13 ), RB( 14 ), RB( 15 ) );
#undef RB

ReverseBits::ReverseBits( uint32_t count ) :
	lookupTable( s_reverseBitsLookup ),
	m_pfn( s_functions[ count ] )
{
	assert( count <= 32 );
}