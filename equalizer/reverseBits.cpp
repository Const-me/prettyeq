#include "stdafx.h"
#include "reverseBits.h"

// Compatibility macros to emit `bswap` instructions
#ifdef _MSC_VER
static inline uint32_t bs16( uint32_t x ) { return _byteswap_ushort( (uint16_t)x ); }
static inline uint32_t bs32( uint32_t x ) { return _byteswap_ulong( x ); }
#else
// Assuming gcc or clang; intel has yet another name, _bswap().
static inline uint32_t bs16( uint32_t x ) { return __builtin_bswap16( (uint16_t)x ); }
static inline uint32_t bs32( uint32_t x ) { return __builtin_bswap32( x ); }
#endif

// The template function that reverses a fixed count of lowest bits in an integer
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
		// Flipping no more than 8 bits, no need to process other bytes
		constexpr int lastShift = ( 8 - ( n % 8 ) ) % 8;
		if constexpr( 0 == lastShift )
			return x;
		return x >> lastShift;
	}
	// Flip bytes, there's an instruction for that, pretty fast

	if constexpr( n == 16 )
	{
		// When flipping exactly 16 bits, we only need to swap 2 bytes, no extra shifts needed then.
		return bs16( x );
	}

	x = bs32( x );
	constexpr int lastShift = ( 32 - n ) % 32;
	if constexpr( 0 == lastShift )
		return x;
	return x >> lastShift;
}

const std::array<ReverseBits::pfn, 33> ReverseBits::s_implementations =
{
#define DF4( i ) &reverseBitsImpl<i>, &reverseBitsImpl<i+1>, &reverseBitsImpl<i+2>, &reverseBitsImpl<i+3>
	DF4( 0 ), DF4( 4 ), DF4( 8 ), DF4( 12 ), DF4( 16 ), DF4( 20 ), DF4( 24 ), DF4( 28 ),
#undef DF4
	&reverseBitsImpl<32>
};