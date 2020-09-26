#include "stdafx.h"
#include "reverseBits.h"

template<int n>
static uint32_t reverseBitsImpl( uint32_t x )
{
	static_assert( n >= 0 && n <= 32 );
	if constexpr( n <= 1 )
		return x;
	// Flip pairwise
	x = ( ( x & 0x55555555 ) << 1 ) | ( ( x & 0xAAAAAAAA ) >> 1 );
	// Flip pairs
	x = ( ( x & 0x33333333 ) << 2 ) | ( ( x & 0xCCCCCCCC ) >> 2 );
	// Flip nibbles
	x = ( ( x & 0x0F0F0F0F ) << 4 ) | ( ( x & 0xF0F0F0F0 ) >> 4 );
	if constexpr( n <= 8 )
	{
		constexpr int lastShift = ( 8 - ( n % 8 ) ) % 8;
		if constexpr( 0 == lastShift )
			return x;
		return x >> lastShift;
	}

	// Flip bytes, there's an instruction for that, pretty fast
	x = _byteswap_ulong( x );
	constexpr int lastShift = ( 32 - n ) % 32;
	if constexpr( 0 == lastShift )
		return x;
	return x >> lastShift;
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

ReverseBits::ReverseBits( uint32_t count ) :
	m_pfn( s_functions[ count ] )
{
	assert( count <= 32 );
}