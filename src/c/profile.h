/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef PROFILE_H
#define PROFILE_H


#ifdef PROFILE
#include <gperftools/profiler.h>
#define PROFILER_START( CSTR )      do{ ProfilerStart(CSTR); }while(0)
#define PROFILER_STOP_START( CSTR ) do{ ProfilerStop(); ProfilerStart(CSTR); }while(0)
#define PROFILER_STOP               ProfilerStop()
#else
#define PROFILER_START( CSTR )      do{}while(0)
#define PROFILER_STOP_START( CSTR ) do{}while(0)
#define PROFILER_STOP               do{}while(0)
#endif

#endif //PROFILE_H
