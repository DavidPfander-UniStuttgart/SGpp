// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

// include within class of OperationMultipleEvalSubspaceCombined directly!
// -> therefore not in namespace

static inline void calculateIndexCombined(
    size_t dim, size_t nextIterationToRecalc,
    const double *const (&dataTuplePtr)[4], std::vector<uint32_t> &hInversePtr,
    uint32_t *(&partialIndicesFlat)[4], double *(&partialPhiEvals)[4],
    // uint32_t *(&indexPtr)[4],
    uint32_t (&indexFlat)[4], double (&phiEval)[4]) {
  __m128i oneIntegerReg = _mm_set1_epi32((uint32_t)1);

  union {
    __m128d doubleRegister;
    __m128i integerRegister;
    uint32_t uint32Value[4];
  } sseUnion;

  // flatten only
  __m128i indexFlatReg =
      _mm_set_epi32(partialIndicesFlat[3][nextIterationToRecalc],
                    partialIndicesFlat[2][nextIterationToRecalc],
                    partialIndicesFlat[1][nextIterationToRecalc],
                    partialIndicesFlat[0][nextIterationToRecalc]);

  // evaluate only
  union {
    __m256d doubleRegister;
    double doubleValue[4];
  } avxUnion;

  int64_t absIMask = 0x7FFFFFFFFFFFFFFF;
  double *fabsMask = (double *)&absIMask;
  __m256d absMask = _mm256_broadcast_sd(fabsMask);
  __m256d one = _mm256_set1_pd(1.0);
  //__m256d zero = _mm256_set1_pd(0.0);

  __m256d phiEvalReg = _mm256_set_pd(partialPhiEvals[3][nextIterationToRecalc],
                                     partialPhiEvals[2][nextIterationToRecalc],
                                     partialPhiEvals[1][nextIterationToRecalc],
                                     partialPhiEvals[0][nextIterationToRecalc]);

  for (size_t i = nextIterationToRecalc; i < dim; i += 1) {
    __m256d dataTupleReg =
        _mm256_set_pd(dataTuplePtr[3][i], dataTuplePtr[2][i],
                      dataTuplePtr[1][i], dataTuplePtr[0][i]);

    __m256d hInverseReg = _mm256_set1_pd((double)hInversePtr[i]);
    __m256d unadjustedReg = _mm256_mul_pd(dataTupleReg, hInverseReg);

    // implies flooring
    __m128i roundedReg = _mm256_cvttpd_epi32(unadjustedReg);
    __m128i andedReg = _mm_and_si128(oneIntegerReg, roundedReg);
    __m128i signReg = _mm_xor_si128(oneIntegerReg, andedReg);
    __m128i indexReg = _mm_add_epi32(roundedReg, signReg);

    // flatten index
    uint32_t actualDirectionGridPoints = hInversePtr[i];
    actualDirectionGridPoints >>= 1;
    __m128i actualDirectionGridPointsReg =
        _mm_set1_epi32(actualDirectionGridPoints);

    indexFlatReg = _mm_mullo_epi32(indexFlatReg, actualDirectionGridPointsReg);

    __m128i indexShiftedReg = _mm_srli_epi32(indexReg, 1);

    indexFlatReg = _mm_add_epi32(indexFlatReg, indexShiftedReg);

    sseUnion.integerRegister = indexFlatReg;
    partialIndicesFlat[0][i + 1] = sseUnion.uint32Value[0];
    partialIndicesFlat[1][i + 1] = sseUnion.uint32Value[1];
    partialIndicesFlat[2][i + 1] = sseUnion.uint32Value[2];
    partialIndicesFlat[3][i + 1] = sseUnion.uint32Value[3];

    // evaluate
    __m256d indexDoubleReg = _mm256_cvtepi32_pd(indexReg);

    __m256d phi1DEvalReg = _mm256_mul_pd(hInverseReg, dataTupleReg);
    phi1DEvalReg = _mm256_sub_pd(phi1DEvalReg, indexDoubleReg);

    phi1DEvalReg = _mm256_and_pd(phi1DEvalReg, absMask);
    phi1DEvalReg = _mm256_sub_pd(one, phi1DEvalReg);
    // phi1DEvalReg = _mm256_max_pd(zero, phi1DEvalReg);

    phiEvalReg = _mm256_mul_pd(phiEvalReg, phi1DEvalReg);

    avxUnion.doubleRegister = phiEvalReg;
    partialPhiEvals[0][i + 1] = avxUnion.doubleValue[0];
    partialPhiEvals[1][i + 1] = avxUnion.doubleValue[1];
    partialPhiEvals[2][i + 1] = avxUnion.doubleValue[2];
    partialPhiEvals[3][i + 1] = avxUnion.doubleValue[3];
  }

  // may a structure ind[0] im[0] eval[0] null ind[1] ... might help
  //}
  _mm_storeu_si128((__m128i *)indexFlat, indexFlatReg);
  _mm256_storeu_pd(phiEval, phiEvalReg);
}

static inline void calculateIndexCombined2(
    size_t dim, size_t nextIterationToRecalc,
    // rep
    const double *const (&dataTuplePtr)[4],
    const double *const (&dataTuplePtr2)[4], std::vector<uint32_t> &hInversePtr,
    // rep
    uint32_t *(&partialIndicesFlat)[4], uint32_t *(&partialIndicesFlat2)[4],
    // rep
    double *(&partialPhiEvals)[4], double *(&partialPhiEvals2)[4],
    // rep
    uint32_t (&indexFlat)[4], uint32_t (&indexFlat2)[4],
    // rep
    double (&phiEval)[4], double (&phiEval2)[4]) {
  __m128i oneIntegerReg = _mm_set1_epi32((uint32_t)1);

  union {
    __m128d doubleRegister;
    __m128i integerRegister;
    uint32_t uint32Value[4];
  } sseUnion;

  // flatten only
  __m128i indexFlatReg =
      _mm_set_epi32(partialIndicesFlat[3][nextIterationToRecalc],
                    partialIndicesFlat[2][nextIterationToRecalc],
                    partialIndicesFlat[1][nextIterationToRecalc],
                    partialIndicesFlat[0][nextIterationToRecalc]);
  __m128i indexFlatReg2 =
      _mm_set_epi32(partialIndicesFlat2[3][nextIterationToRecalc],
                    partialIndicesFlat2[2][nextIterationToRecalc],
                    partialIndicesFlat2[1][nextIterationToRecalc],
                    partialIndicesFlat2[0][nextIterationToRecalc]);

  // evaluate only
  union {
    __m256d doubleRegister;
    double doubleValue[4];
  } avxUnion;

  int64_t absIMask = 0x7FFFFFFFFFFFFFFF;
  double *fabsMask = (double *)&absIMask;
  __m256d absMask = _mm256_broadcast_sd(fabsMask);
  __m256d one = _mm256_set1_pd(1.0);
  __m256d zero = _mm256_set1_pd(0.0);

  __m256d phiEvalReg = _mm256_set_pd(partialPhiEvals[3][nextIterationToRecalc],
                                     partialPhiEvals[2][nextIterationToRecalc],
                                     partialPhiEvals[1][nextIterationToRecalc],
                                     partialPhiEvals[0][nextIterationToRecalc]);
  __m256d phiEvalReg2 =
      _mm256_set_pd(partialPhiEvals2[3][nextIterationToRecalc],
                    partialPhiEvals2[2][nextIterationToRecalc],
                    partialPhiEvals2[1][nextIterationToRecalc],
                    partialPhiEvals2[0][nextIterationToRecalc]);

  for (size_t i = nextIterationToRecalc; i < dim; i += 1) {
    __m256d dataTupleReg =
        _mm256_set_pd(dataTuplePtr[3][i], dataTuplePtr[2][i],
                      dataTuplePtr[1][i], dataTuplePtr[0][i]);
    __m256d dataTupleReg2 =
        _mm256_set_pd(dataTuplePtr2[3][i], dataTuplePtr2[2][i],
                      dataTuplePtr2[1][i], dataTuplePtr2[0][i]);

    __m256d hInverseReg = _mm256_set1_pd((double)hInversePtr[i]);

    __m256d unadjustedReg = _mm256_mul_pd(dataTupleReg, hInverseReg);
    __m256d unadjustedReg2 = _mm256_mul_pd(dataTupleReg2, hInverseReg);

    // implies flooring
    __m128i roundedReg = _mm256_cvttpd_epi32(unadjustedReg);
    __m128i roundedReg2 = _mm256_cvttpd_epi32(unadjustedReg2);

    __m128i andedReg = _mm_and_si128(oneIntegerReg, roundedReg);
    __m128i andedReg2 = _mm_and_si128(oneIntegerReg, roundedReg2);

    __m128i signReg = _mm_xor_si128(oneIntegerReg, andedReg);
    __m128i signReg2 = _mm_xor_si128(oneIntegerReg, andedReg2);

    __m128i indexReg = _mm_add_epi32(roundedReg, signReg);
    __m128i indexReg2 = _mm_add_epi32(roundedReg2, signReg2);

    // flatten index
    uint32_t actualDirectionGridPoints = hInversePtr[i];
    actualDirectionGridPoints >>= 1;
    __m128i actualDirectionGridPointsReg =
        _mm_set1_epi32(actualDirectionGridPoints);

    indexFlatReg = _mm_mullo_epi32(indexFlatReg, actualDirectionGridPointsReg);
    indexFlatReg2 =
        _mm_mullo_epi32(indexFlatReg2, actualDirectionGridPointsReg);

    __m128i indexShiftedReg = _mm_srli_epi32(indexReg, 1);
    __m128i indexShiftedReg2 = _mm_srli_epi32(indexReg2, 1);

    indexFlatReg = _mm_add_epi32(indexFlatReg, indexShiftedReg);
    indexFlatReg2 = _mm_add_epi32(indexFlatReg2, indexShiftedReg2);

    sseUnion.integerRegister = indexFlatReg;
    partialIndicesFlat[0][i + 1] = sseUnion.uint32Value[0];
    partialIndicesFlat[1][i + 1] = sseUnion.uint32Value[1];
    partialIndicesFlat[2][i + 1] = sseUnion.uint32Value[2];
    partialIndicesFlat[3][i + 1] = sseUnion.uint32Value[3];

    sseUnion.integerRegister = indexFlatReg2;
    partialIndicesFlat2[0][i + 1] = sseUnion.uint32Value[0];
    partialIndicesFlat2[1][i + 1] = sseUnion.uint32Value[1];
    partialIndicesFlat2[2][i + 1] = sseUnion.uint32Value[2];
    partialIndicesFlat2[3][i + 1] = sseUnion.uint32Value[3];

    // evaluate
    __m256d indexDoubleReg = _mm256_cvtepi32_pd(indexReg);
    __m256d indexDoubleReg2 = _mm256_cvtepi32_pd(indexReg2);

    __m256d phi1DEvalReg = _mm256_mul_pd(hInverseReg, dataTupleReg);
    __m256d phi1DEvalReg2 = _mm256_mul_pd(hInverseReg, dataTupleReg2);

    phi1DEvalReg = _mm256_sub_pd(phi1DEvalReg, indexDoubleReg);
    phi1DEvalReg2 = _mm256_sub_pd(phi1DEvalReg2, indexDoubleReg2);

    phi1DEvalReg = _mm256_and_pd(phi1DEvalReg, absMask);
    phi1DEvalReg2 = _mm256_and_pd(phi1DEvalReg2, absMask);

    phi1DEvalReg = _mm256_sub_pd(one, phi1DEvalReg);
    phi1DEvalReg2 = _mm256_sub_pd(one, phi1DEvalReg2);

    phi1DEvalReg = _mm256_max_pd(zero, phi1DEvalReg);
    phi1DEvalReg2 = _mm256_max_pd(zero, phi1DEvalReg2);

    phiEvalReg = _mm256_mul_pd(phiEvalReg, phi1DEvalReg);
    phiEvalReg2 = _mm256_mul_pd(phiEvalReg2, phi1DEvalReg2);

    avxUnion.doubleRegister = phiEvalReg;
    partialPhiEvals[0][i + 1] = avxUnion.doubleValue[0];
    partialPhiEvals[1][i + 1] = avxUnion.doubleValue[1];
    partialPhiEvals[2][i + 1] = avxUnion.doubleValue[2];
    partialPhiEvals[3][i + 1] = avxUnion.doubleValue[3];

    avxUnion.doubleRegister = phiEvalReg2;
    partialPhiEvals2[0][i + 1] = avxUnion.doubleValue[0];
    partialPhiEvals2[1][i + 1] = avxUnion.doubleValue[1];
    partialPhiEvals2[2][i + 1] = avxUnion.doubleValue[2];
    partialPhiEvals2[3][i + 1] = avxUnion.doubleValue[3];
  }

  // may a structure ind[0] im[0] eval[0] null ind[1] ... might help
  _mm_storeu_si128((__m128i *)indexFlat, indexFlatReg);
  _mm_storeu_si128((__m128i *)indexFlat2, indexFlatReg2);

  _mm256_storeu_pd(phiEval, phiEvalReg);
  _mm256_storeu_pd(phiEval2, phiEvalReg2);
}
