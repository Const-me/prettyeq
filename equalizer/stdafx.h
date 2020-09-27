#pragma once
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#define __forceinline inline __attribute__((always_inline))
#endif

#include <stdio.h>
#include <assert.h>
#include <array>
#include <vector>
#include <algorithm>