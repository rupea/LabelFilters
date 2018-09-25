/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef __MUTEXLOCK_H
#define __MUTEXLOCK_H


#ifdef _OPENMP
# include <omp.h>
struct MutexType
{
  MutexType() { omp_init_lock(&lock); }
  ~MutexType() { omp_destroy_lock(&lock); }
  void Lock() { omp_set_lock(&lock); }
  void Unlock() { omp_unset_lock(&lock); }
  bool TestLock() { return omp_test_lock(&lock); }
  void YieldLock() 
  {
    while (!omp_test_lock(&lock)){
      #pragma omp taskyield
    }
  }

  MutexType(const MutexType& ) { omp_init_lock(&lock); }
  MutexType& operator= (const MutexType& ) { return *this; }
public:
  omp_lock_t lock;
};
#else
/* A dummy mutex that doesn't actually exclude anything,
 * but as there is no parallelism either, no worries. */
struct MutexType
{
  void Lock() {}
  void Unlock() {}
  bool TestLock() {return true;}
  void YieldLock() {}
};
#endif

/* An exception-safe scoped lock-keeper. */
struct ScopedLock
{
  explicit ScopedLock(MutexType& m, bool yield) : mut(m), locked(true) 
  {
    if (yield) 
      mut.YieldLock(); 
    else 
      mut.Lock();
  }
  ~ScopedLock() { Unlock(); }
  void Unlock() { if(!locked) return; locked=false; mut.Unlock(); }
  void LockAgain() { if(locked) return; mut.Lock(); locked=true; }
private:
  MutexType& mut;
  bool locked;
private: // prevent copying the scoped lock.
   void operator=(const ScopedLock&);
  ScopedLock(const ScopedLock&);
};

#endif
