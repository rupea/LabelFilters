#!/bin/bash
export OMP_DISPLAY_ENV=VERBOSE
{ /usr/bin/time -v lua_cpp solve-quad2.lua; } >& mnist-q2a.log
