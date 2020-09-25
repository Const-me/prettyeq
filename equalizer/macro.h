#pragma once

#define PRETTY_EXPORT __attribute__ ((visibility ("default")))
#define MAY_ALIAS __attribute__((__may_alias__))
#define _likely_(x)      __builtin_expect(!!(x), 1)
#define _unlikely_(x)    __builtin_expect(!!(x), 0)
