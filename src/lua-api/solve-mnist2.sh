#!/bin/bash
export OMP_DISPLAY_ENV=VERBOSE
{ /usr/bin/time -v lua_cpp solve-mnist2.lua; } 2>&1 | tee mnist2i.log
