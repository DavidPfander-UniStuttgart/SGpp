// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

/*
 * Don't remove the "ifndef", they are required to overwrite the parameters though compiler's "-D"
 *
 */


// number of data elements processed in parallel
// should be divisible by vector size
// corresponds to data chunk size
#ifndef SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS
//#define SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS 4
#define SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS 256
//#define SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS 192
#endif

#ifndef SUBSPACEAUTOTUNETMP_ENABLE_SUBSPACE_SKIPPING
#define SUBSPACEAUTOTUNETMP_ENABLE_SUBSPACE_SKIPPING 1
#endif

#ifndef SUBSPACEAUTOTUNETMP_UNROLL
#define SUBSPACEAUTOTUNETMP_UNROLL 0
// if set to 1, implies SUBSPACEAUTOTUNETMP_VEC_PADDING == 8
#endif

#ifndef SUBSPACEAUTOTUNETMP_VEC_PADDING
#define SUBSPACEAUTOTUNETMP_VEC_PADDING 4
//#define SUBSPACEAUTOTUNETMP_VEC_PADDING 8
//#define SUBSPACEAUTOTUNETMP_VEC_PADDING 24
#endif

#ifndef SUBSPACEAUTOTUNETMP_STREAMING_THRESHOLD
// good value: #define SUBSPACEAUTOTUNETMP_STREAMING_THRESHOLD 128
#define SUBSPACEAUTOTUNETMP_STREAMING_THRESHOLD 128
#endif

#ifndef SUBSPACEAUTOTUNETMP_LIST_RATIO
// good value: #define SUBSPACEAUTOTUNETMP_LIST_RATIO 0.1
#define SUBSPACEAUTOTUNETMP_LIST_RATIO 0.2
#endif

#ifndef SUBSPACEAUTOTUNETMP_ENABLE_PARTIAL_RESULT_REUSAGE
#define SUBSPACEAUTOTUNETMP_ENABLE_PARTIAL_RESULT_REUSAGE 1
#endif

// only set from the outside
//#define SUBSPACEAUTOTUNETMP_WRITE_STATS "stats.out"
