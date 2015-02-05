// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef SGPP_OPTIMIZATION_OPTIMIZER_NELDERMEAD_HPP
#define SGPP_OPTIMIZATION_OPTIMIZER_NELDERMEAD_HPP

#include <sgpp/globaldef.hpp>

#include <sgpp/optimization/optimizer/Optimizer.hpp>

namespace SGPP {
  namespace optimization {
    namespace optimizer {

      /**
       * Gradient-free Nelder-Mead method.
       */
      class NelderMead : public Optimizer {
        public:
          /// default reflection coefficient
          static const float_t DEFAULT_ALPHA;
          /// default expansion coefficient
          static const float_t DEFAULT_BETA;
          /// default contraction coefficient
          static const float_t DEFAULT_GAMMA;
          /// default shrinking coefficient
          static const float_t DEFAULT_DELTA;
          /// default maximal number of function evaluations
          static const size_t DEFAULT_MAX_FCN_EVAL_COUNT = 1000;
          /// edge length of starting simplex
          static const float_t STARTING_SIMPLEX_EDGE_LENGTH;

          /**
           * Constructor.
           * The starting point is set to \f$(0.5, \dotsc, 0.5)^{\mathrm{T}}\f$.
           *
           * @param f                     objective function
           * @param maxFcnEvalCount       maximal number of function evaluations
           * @param alpha                 reflection coefficient
           * @param beta                  expansion coefficient
           * @param gamma                 contraction coefficient
           * @param delta                 shrinking coefficient
           */
          NelderMead(const function::Objective& f,
                     size_t maxFcnEvalCount = DEFAULT_MAX_FCN_EVAL_COUNT,
                     float_t alpha = DEFAULT_ALPHA,
                     float_t beta = DEFAULT_BETA,
                     float_t gamma = DEFAULT_GAMMA,
                     float_t delta = DEFAULT_DELTA);

          /**
           * @param[out] xOpt optimal point
           * @return          optimal objective function value
           */
          float_t optimize(std::vector<float_t>& xOpt);

          /**
           * @return pointer to cloned object
           */
          Optimizer* clone();

          /**
           * @return          reflection coefficient
           */
          float_t getAlpha() const;

          /**
           * @param alpha     reflection coefficient
           */
          void setAlpha(float_t alpha);

          /**
           * @return          expansion coefficient
           */
          float_t getBeta() const;

          /**
           * @param beta      expansion coefficient
           */
          void setBeta(float_t beta);

          /**
           * @return          contraction coefficient
           */
          float_t getGamma() const;

          /**
           * @param gamma     contraction coefficient
           */
          void setGamma(float_t gamma);

          /**
           * @return          shrinking coefficient
           */
          float_t getDelta() const;

          /**
           * @param delta     shrinking coefficient
           */
          void setDelta(float_t delta);

        protected:
          /// reflection coefficient
          float_t alpha;
          /// expansion coefficient
          float_t beta;
          /// contraction coefficient
          float_t gamma;
          /// shrinking coefficient
          float_t delta;
      };

    }
  }
}

#endif
