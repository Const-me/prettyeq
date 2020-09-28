#include "stdafx.h"
#include "fft.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include "Stopwatch.hpp"

static const float single_channel_5000HZ_data[] = {
	-0.120544, 0.222534, 0.457336, 0.469727, 0.253632, -0.085846, -0.383575,
	-0.494568, -0.365295, -0.058197, 0.277222, 0.477783, 0.445923, 0.197205,
	-0.147461, -0.420441, -0.488922, -0.319580, 0.005157, 0.327362, 0.490356,
	0.414825, 0.137512, -0.206696, -0.450378, -0.475037, -0.268646, 0.068390,
	0.372162, 0.494537, 0.376892, 0.075592, -0.262482, -0.472900, -0.453339,
	-0.213257, 0.130524, 0.410797, 0.491302, 0.332794, 0.012451, -0.313995,
	-0.487701, -0.424225, -0.154419, 0.190491, 0.442719, 0.479614, 0.283234,
	-0.050934, -0.360352, -0.494476, -0.388123, -0.093018, 0.247345, 0.467377,
	0.460083, 0.229004, -0.113464, -0.400787, -0.493164, -0.345673, -0.030090,
	0.300110, 0.484344, 0.432983, 0.171021, -0.174133, -0.434601, -0.483704,
	-0.297546, 0.033325, 0.347992, 0.493378, 0.398773, 0.110199, -0.231964,
	-0.461334, -0.466339, -0.244507, 0.096191, 0.390137, 0.494293, 0.358032,
	0.047607, -0.285980, -0.480469, -0.441284, -0.187500, 0.157501, 0.425873,
	0.487091, 0.311401, -0.015778, -0.335266, -0.491730, -0.409027, -0.127380,
	0.216217, 0.454590, 0.471893, 0.259644, -0.078888, -0.379059, -0.494568,
	-0.370026, -0.065186, 0.271332, 0.475891, 0.448944, 0.203644, -0.140717,
	-0.416687, -0.489960, -0.324951, -0.001892, 0.322052, 0.489349, 0.418640,
	0.144287, -0.200256, -0.447388, -0.476959, -0.274536, 0.061401, 0.367462,
	0.494537, 0.381439, 0.082550, -0.256500, -0.470795, -0.456116, -0.219604,
	0.123688, 0.406830, 0.492065, 0.337982, 0.019470, -0.308502, -0.486450,
	-0.427826, -0.161102, 0.183960, 0.439545, 0.481293, 0.288971, -0.043915,
	-0.355469, -0.494141, -0.392487, -0.099945, 0.241211, 0.465027, 0.462616,
	0.235229, -0.106598, -0.396576, -0.493683, -0.350677, -0.037109, 0.294495,
	0.482849, 0.436340, 0.177612, -0.167542, -0.431213, -0.485168, -0.303131,
	0.026276, 0.342926, 0.492767, 0.402924, 0.117065, -0.225708, -0.458740,
	-0.468628, -0.250610, 0.089264, 0.385742, 0.494537, 0.362854, 0.054626,
	-0.280182, -0.478729, -0.444458, -0.194000, 0.150787, 0.422241, 0.488281,
	0.316864, -0.008728, -0.330048, -0.490845, -0.412933, -0.134186, 0.209839,
	0.451782, 0.473969, 0.265625, -0.071930, -0.374512, -0.494568, -0.374664,
	-0.072174, 0.265411, 0.473877, 0.451874, 0.210022, -0.133942, -0.412811,
	-0.490906, -0.330231, -0.008972, 0.316650, 0.488251, 0.422363, 0.151031,
	-0.193787, -0.444336, -0.478790, -0.280365, 0.054382, 0.362671, 0.494537,
	0.385895, 0.089508, -0.250427, -0.468567, -0.458801, -0.225922, 0.116852,
	0.402802, 0.492798, 0.343109, 0.026520, -0.302948, -0.485107, -0.431305,
	-0.167755, 0.177399, 0.436249, 0.482910, 0.294678, -0.036896, -0.350525,
	-0.493683, -0.396729, -0.106842, 0.235016, 0.462555, 0.465088, 0.241425,
	-0.099701, -0.392303, -0.494141, -0.355621, -0.044159, 0.288788, 0.481262,
	0.439636, 0.184174, -0.160858, -0.427704, -0.486511, -0.308685, 0.019257,
	0.337830, 0.492065, 0.406982, 0.123932, -0.219421, -0.456024, -0.470856,
	-0.256683, 0.082306, 0.381287, 0.494537, 0.367615, 0.061646, -0.274323,
	-0.476868, -0.447479, -0.200470, 0.144043, 0.418518, 0.489380, 0.322235,
	-0.001678, -0.324768, -0.489899, -0.416779, -0.140961, 0.203400, 0.448822,
	0.475952, 0.271545, -0.064941, -0.369873, -0.494568, -0.379242, -0.079132,
	0.259430, 0.471832, 0.454712, 0.216400, -0.127167, -0.408875, -0.491760,
	-0.335449, -0.016022, 0.311218, 0.487061, 0.425995, 0.157715, -0.187286,
	-0.441193, -0.480530, -0.286163, 0.047363, 0.357849, 0.494293, 0.390289,
	0.096436, -0.244324, -0.466248, -0.461395, -0.232178, 0.109985, 0.398651,
	0.493378, 0.348145, 0.033569, -0.297333, -0.483673, -0.434723, -0.174347,
	0.170807, 0.432861, 0.484406, 0.300323, -0.029846, -0.345520, -0.493134,
	-0.400909, -0.113708, 0.228790, 0.459991, 0.467438, 0.247528, -0.092773,
	-0.388000, -0.494507, -0.360504, -0.051178, 0.283020, 0.479584, 0.442841,
	0.190704, -0.154175, -0.424103, -0.487762, -0.314178, 0.012207, 0.332611,
	0.491241, 0.410950, 0.130737, -0.213074, -0.453247, -0.472992, -0.262695,
	0.075348, 0.376740, 0.494537, 0.372314, 0.068634, -0.268433, -0.474945,
	-0.450470, -0.206909, 0.137299, 0.414703, 0.490387, 0.327545, 0.005402,
	-0.319397, -0.488861, -0.420563, -0.147705, 0.196991, 0.445862, 0.477844,
	0.277435, -0.057953, -0.365143, -0.494568, -0.383728, -0.086090, 0.253418,
	0.469635, 0.457428, 0.222717, -0.120331, -0.404877, -0.492493, -0.340607,
	-0.023071, 0.305695, 0.485748, 0.429535, 0.164398, -0.180725, -0.437958,
	-0.482178, -0.291870, 0.040344, 0.352966, 0.493866, 0.394562, 0.103333,
	-0.238159, -0.463837, -0.463928, -0.238373, 0.103088, 0.394440, 0.493896,
	0.353119, 0.040588, -0.291687, -0.482117, -0.438049, -0.180939, 0.164154,
	0.429413, 0.485809, 0.305878, -0.022797, -0.340424, -0.492462, -0.404999,
	-0.120544, 0.222504, 0.457336, 0.469727, 0.253632, -0.085846, -0.383575,
	-0.494568, -0.365295, -0.058197, 0.277222, 0.477783, 0.445923, 0.197205,
	-0.147461, -0.420410, -0.488922, -0.319580, 0.005127, 0.327362, 0.490356,
	0.414825, 0.137512, -0.206696, -0.450348, -0.475037, -0.268646, 0.068390,
	0.372131, 0.494537, 0.376892, 0.075592, -0.262482, -0.472900, -0.453339,
	-0.213287, 0.130524, 0.410797, 0.491302, 0.332794, 0.012451, -0.313995,
	-0.487701, -0.424225, -0.154419, 0.190491, 0.442719, 0.479645, 0.283234,
	-0.050934, -0.360352, -0.494476, -0.388153, -0.093018, 0.247345, 0.467377,
	0.460083, 0.229004, -0.113464, -0.400757, -0.493164, -0.345673, -0.030090,
	0.300110, 0.484344, 0.432983, 0.171021, -0.174133, -0.434601, -0.483704,
	-0.297546,
};

static const float dual_channel_micro[] = {
	-0.120544, -0.120544, 0.222534, 0.222534,
};

struct TestCases
{
	int frequency;
	float expected_psd;
};

TestCases cases[] = {
	{
		100,
		0.09595596265773097,
	},
	{
		200,
		0.09558897341820413,
	},
	{
		1000,
		0.10730647827342488,
	},
	{
		2000,
		0.14449811458513645,
	},
	{
		5000,
		126.14145371375102,
	},
	{
		10000,
		0.14292403328440603,
	},
};

static inline bool float_approx_equal( float f1, float f2 ) 
{
	// Not only FMA is faster, it's also more precise because only rounds once.
	// Emulating FMA slightly decreases the precision of sin/cos code which computes the lookup table.
#if defined( EMULATED_FMA ) && ( !FFT_PRECISE_TABLE )
	constexpr float precision = 0.00003;
#else
	constexpr float precision = 0.00001;
#endif
	return fabsf( f1 - f2 ) > precision ? false : true;
}

void test_init()
{
	Stopwatch sw;
	fft_init();
	printf( "[FFT Init %d samples] time: %lf ms\n", MAX_SAMPLES, sw.elapsedMilliseconds() );
}

void test_single_channel()
{
	complex output[ MAX_SAMPLES ];
	unsigned int N = sizeof( single_channel_5000HZ_data ) / sizeof( single_channel_5000HZ_data[ 0 ] );

	Stopwatch sw;
	fft_run( single_channel_5000HZ_data, output, N, 1 );
	const double us = sw.elapsedMicroseconds();

	for( int i = 0; i < sizeof( cases ) / sizeof( cases[ 0 ] ); i++ ) 
	{
		const int sample_index = FFT_FREQ_TO_SAMPLE( N, cases[ i ].frequency );
		const float psd = FFT_PSD( output[ sample_index ] );
		const bool equals = float_approx_equal( cases[ i ].expected_psd, psd );
		if( equals )
			continue;
		printf( "Test failed\n" );
	}
	printf( "[FFT Run %d samples] time: %lf us\n", N, us );
}

void test_dual_channel_micro() 
{
	complex output[ MAX_SAMPLES ];
	unsigned int N = sizeof( dual_channel_micro ) / sizeof( dual_channel_micro[ 0 ] );

	Stopwatch sw;
	fft_run( dual_channel_micro, output, N, 2 );
	printf( "[FFT Run %d samples] time: %lf us\n", N, sw.elapsedMicroseconds() );
}

int main( int argc, const char **argv )
{
	test_init();
	test_single_channel();
	test_dual_channel_micro();
	return 0;
}