#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    defines.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, Université Côte d'Azur, LEAT, France
  * @version 2.1.0
  * @date    10 january 2024
  * @brief   Global C pre-processor definitions to use to build all source files (incl. CMSIS-NN)
  */

/* CMSIS-NN round mode definition */
#if defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)


#define ARM_NN_TRUNCATE 1
#define RISCV_NN_TRUNCATE 1

#endif // defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef TRAPV_SHIFT
#include <limits.h>
#include <stdio.h>
#include <assert.h>
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor, round_mode) scale_number_t_ ## type (number, scale_factor, round_mode)
#define scale(type, number, scale_factor, round_mode) _scale(type, number, scale_factor, round_mode)
#define _scale_and_clamp_to(type, number, scale_factor, round_mode) scale_and_clamp_to_number_t_ ## type (number, scale_factor, round_mode)
#define scale_and_clamp_to(type, number, scale_factor, round_mode) _scale_and_clamp_to(type, number, scale_factor, round_mode)

typedef enum {
  ROUND_MODE_NONE,
  ROUND_MODE_FLOOR,
  ROUND_MODE_NEAREST,
} round_mode_t;

// Idea 1: Write the smallest min max interval of the net, could be an issue for hybrid int type network
// Idea 2: listing any interval and add type in name in a switch case like <- better but painfull
// #define NUMBER_MIN		// Max value for this numeric type
// #define NUMBER_MAX		// Min value for this numeric type

// // Idea 1: List of all types and write any corresponding function 
// typedef  number_t;		// Standard size numeric type used for weights and activations
// typedef  long_number_t;	// Long numeric type used for intermediate results

#define NUMBER_MIN_INT16_T -32768
#define NUMBER_MAX_INT16_T 32767

static inline int32_t min_int16_t(
    int32_t a,
    int32_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int32_t max_int16_t(
    int32_t a,
    int32_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int32_t scale_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT32_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%d, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT32_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int16_t clamp_to_number_t_int16_t(
  int32_t number) {
	return (int16_t) max_int16_t(
      NUMBER_MIN_INT16_T,
      min_int16_t(
        NUMBER_MAX_INT16_T, number));
}
static inline int16_t scale_and_clamp_to_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int16_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int16_t) * 8);
  }
#else
  number = scale_number_t_int16_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int16_t(number);
#endif
}

#define NUMBER_MIN_INT32_T -2147483648
#define NUMBER_MAX_INT32_T 2147483647

static inline int64_t min_int32_t(
    int64_t a,
    int64_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int64_t max_int32_t(
    int64_t a,
    int64_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int64_t scale_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT64_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%ld, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT64_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int32_t clamp_to_number_t_int32_t(
  int64_t number) {
	return (int32_t) max_int32_t(
      NUMBER_MIN_INT32_T,
      min_int32_t(
        NUMBER_MAX_INT32_T, number));
}
static inline int32_t scale_and_clamp_to_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int32_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int32_t) * 8);
  }
#else
  number = scale_number_t_int32_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int32_t(number);
#endif
}




static inline void int64_t_to_float(int64_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int32_t_to_float(int32_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int16_t_to_float(int16_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}

static inline void int8_t_to_float(int8_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}
#endif //__NUMBER_H__

#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_337_H_
#define _CONV1D_337_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       500
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    7
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_337_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_337(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_337_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_337.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       500
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    7
#define CONV_STRIDE         2
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_337(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    1
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  7
#define CONV_GROUPS       1


const int16_t  conv1d_337_bias[CONV_FILTERS] = {1, -2, 0, -2, -3, 2, 1, -1}
;

const int16_t  conv1d_337_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-14}
, {-18}
, {-1}
, {4}
, {4}
, {33}
, {-29}
}
, {{-4}
, {5}
, {-18}
, {2}
, {18}
, {-37}
, {37}
}
, {{21}
, {-33}
, {27}
, {20}
, {35}
, {-10}
, {9}
}
, {{-2}
, {-22}
, {30}
, {37}
, {4}
, {0}
, {18}
}
, {{-32}
, {5}
, {-21}
, {16}
, {27}
, {-42}
, {11}
}
, {{34}
, {-22}
, {-18}
, {-30}
, {36}
, {-4}
, {-10}
}
, {{34}
, {0}
, {8}
, {-32}
, {-8}
, {4}
, {-26}
}
, {{27}
, {28}
, {15}
, {14}
, {-34}
, {2}
, {14}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_262_H_
#define _BATCH_NORMALIZATION_262_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       247

typedef int16_t batch_normalization_262_output_type[247][8];

#if 0
void batch_normalization_262(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_262_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_262_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_262.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       247
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_262(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_262_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_262_bias[8] = {2, 7, -3, 3, 9, 1, 2, 3}
;
const int16_t batch_normalization_262_kernel[8] = {283, 415, 281, 267, 302, 248, 277, 220}
;
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_92_H_
#define _MAX_POOLING1D_92_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   247
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_92_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_92(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_92_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_92.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   247
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d_92(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_338_H_
#define _CONV1D_338_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       123
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_338_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_338(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_338_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_338.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       123
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_338(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_338_bias[CONV_FILTERS] = {0, 1, -3, -1, -3, 0, -1, -1, -2, -1, -1, -1, -2, 0, -2, -1}
;

const int16_t  conv1d_338_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-22, -37, 12, 30, 24, 18, -34, 19}
, {-19, -10, -13, -8, -15, -31, -7, 22}
, {-2, 34, 0, -10, -7, 6, 14, 31}
}
, {{-27, 34, 24, 11, -26, -16, -2, -1}
, {-19, 35, 27, -19, 12, -3, -22, -13}
, {22, 3, -26, -1, -37, 6, 16, -19}
}
, {{-34, -7, -18, -26, 19, -18, -25, 0}
, {-6, -12, -37, -19, 7, 0, 34, 16}
, {33, -3, 18, 2, -7, 25, 34, 2}
}
, {{30, -16, -20, 18, 14, -1, 14, -29}
, {-32, -25, -26, -23, 0, -18, 17, 25}
, {-23, 18, -8, -23, -20, 6, 0, 22}
}
, {{-14, 5, -28, 12, -26, -36, 29, 14}
, {6, -35, 33, -30, -21, -17, -12, 17}
, {-13, -10, -7, -3, -23, -18, 22, -15}
}
, {{-25, 2, 2, -25, 10, -20, 13, 15}
, {2, -37, -32, 3, 21, -32, 24, 18}
, {6, -20, -1, -37, 33, -6, -29, 1}
}
, {{-6, -34, -5, -1, -10, 21, 7, 32}
, {15, 15, -28, -11, -27, 11, -8, -4}
, {19, -29, -13, -30, -34, -29, 16, -10}
}
, {{5, 38, 20, 27, -16, -33, -21, 21}
, {2, 5, 14, -2, -7, -21, -16, 11}
, {-3, -4, 7, 16, -35, -23, 23, 30}
}
, {{35, -26, -2, -32, 10, -28, 6, 5}
, {33, -5, -27, -21, -1, -11, 10, -12}
, {-35, -6, -4, 13, 12, -11, -24, -17}
}
, {{29, 33, -14, -8, 19, -27, 12, 6}
, {-27, 13, 22, -21, -10, 21, 35, -29}
, {-33, -36, -31, -8, -31, 21, 13, 22}
}
, {{-14, -35, -13, -6, 34, 13, 29, -36}
, {1, 0, 27, 14, -22, 2, -17, -28}
, {-21, -3, -34, 20, 32, 2, -18, 13}
}
, {{21, -23, 26, 31, 2, -6, 4, -12}
, {11, -9, 12, 31, -4, -15, 26, -32}
, {-18, 8, 33, -38, -34, 21, -30, 25}
}
, {{-23, -4, -2, 5, -1, 12, -3, -35}
, {0, 12, 24, -29, -31, 22, -26, 19}
, {-11, -34, -34, -3, -11, 16, 2, -30}
}
, {{-18, -25, -12, 17, 12, -32, 5, 20}
, {10, -15, 34, -18, 23, 21, 27, 20}
, {-1, 39, -16, -1, -13, 5, 10, 4}
}
, {{14, 14, -18, -33, -21, -34, 15, -17}
, {32, 27, -37, -9, 26, 4, 21, -9}
, {-35, 14, 31, 1, -24, 2, -25, -15}
}
, {{20, -23, -29, 27, 36, -8, 8, 12}
, {-21, -12, -36, -10, -35, -36, -4, -23}
, {27, 32, -36, 34, 31, -6, 28, 28}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_263_H_
#define _BATCH_NORMALIZATION_263_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       121

typedef int16_t batch_normalization_263_output_type[121][16];

#if 0
void batch_normalization_263(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_263_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_263_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_263.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       121
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_263(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_263_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_263_bias[16] = {-13, 25, 18, 60, 80, 69, 85, -31, 89, 17, 49, -45, 97, -79, 62, -16}
;
const int16_t batch_normalization_263_kernel[16] = {305, 245, 317, 206, 143, 198, 162, 227, 175, 236, 278, 259, 160, 188, 244, 251}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_339_H_
#define _CONV1D_339_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       121
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_339_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_339(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_339_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_339.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       121
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_339(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_339_bias[CONV_FILTERS] = {-1, -2, 0, -1, 0, 0, -1, -1, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -2}
;

const int16_t  conv1d_339_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0, -2, -9, -14, -7, -29, 20, 19, -13, 22, -5, 15, -34, -19, 5, 24}
, {22, -24, -3, -10, -1, -15, -9, -14, -16, 6, -10, -3, 10, 16, 0, -5}
, {-18, 14, -5, 3, 0, -6, 25, -21, -18, -11, 8, -23, -22, 8, -10, -5}
}
, {{0, 12, -10, 21, 27, -6, 6, 0, 12, -8, 24, -18, 1, -22, 13, 1}
, {11, -8, -2, -1, 18, -12, -19, 1, 25, -21, -1, -24, 1, -8, -7, -15}
, {22, -8, -9, 14, 19, -17, -15, -12, -27, -4, -25, 0, 0, 3, -11, -20}
}
, {{-5, 12, -6, -24, 17, 16, 25, -27, -1, 11, -19, -9, -17, 16, 15, -23}
, {16, 12, -17, 2, 10, -5, 7, -8, 9, 12, -12, -27, 20, 22, -10, 12}
, {-23, -8, -10, -20, -13, 20, 16, 13, -4, 24, 21, -20, 22, 9, 21, 10}
}
, {{15, 16, 5, -28, 3, -3, -26, -7, 17, -10, 9, 12, 20, 0, -20, -7}
, {-18, 21, 0, 17, -26, -5, -7, -10, -9, 13, 11, 19, 21, -25, 10, 11}
, {-2, -17, -9, 1, 4, 8, -27, -20, 16, 25, 2, 3, -12, -14, 12, -16}
}
, {{-19, -6, -15, -28, 3, 17, -22, 23, 15, 9, -7, 3, -20, -25, -15, -5}
, {27, -22, -22, 12, 13, 14, 18, 24, -20, 6, 26, 19, 7, -21, 11, -9}
, {-20, -9, 21, -8, 7, -5, 6, 14, -28, 18, -27, 21, -23, -8, -3, -1}
}
, {{-2, -25, -10, -20, 26, 14, -5, 21, 23, -9, -26, 2, -13, 22, -25, 3}
, {16, 15, -8, -26, -1, -25, -5, -11, -13, -17, -19, 0, -21, -20, -19, -2}
, {-3, 24, 19, 14, -20, 0, -7, -1, -22, 22, -13, -25, 17, 3, 2, 11}
}
, {{-28, 22, -10, -1, 22, 1, 19, 3, 3, -25, 16, 2, -5, -28, -10, -8}
, {-10, 16, 15, 23, -9, -4, 9, 7, 1, 9, -11, 19, -26, 10, 19, 23}
, {18, -11, 12, 0, -12, 12, -19, 6, 17, 8, -4, -28, -8, 7, -5, -17}
}
, {{-8, -16, -9, 17, 10, -23, -27, -22, -4, -14, 26, 22, -7, 21, 10, 23}
, {13, -20, -17, -8, 9, 7, 18, 26, 10, -7, 25, 10, -3, -10, 12, 16}
, {21, 20, -10, -22, 22, 14, 10, -11, 7, 20, -15, -12, 6, 4, 20, -24}
}
, {{-20, -7, -23, -27, -18, 17, 16, -14, -22, -1, -8, 0, -3, 18, -12, 3}
, {-25, 3, -3, -13, 19, 9, 14, 4, -17, 3, 12, -20, -16, 13, 14, 19}
, {-14, -13, -3, -14, -25, -25, 2, -7, -1, -8, 3, 23, -24, -8, 16, -17}
}
, {{-24, -29, -13, 7, 4, 12, 2, 14, -23, -9, 0, -17, -22, -10, -6, 11}
, {21, -16, -7, -20, -15, -2, -2, 13, 23, 13, 9, 5, -1, 23, 13, -7}
, {3, -25, -14, -17, -16, -11, 1, 20, 15, 19, 4, 25, 10, 9, -10, 24}
}
, {{5, -12, -20, -6, -1, 14, 2, 11, -4, -9, -13, 19, -6, 17, -23, -11}
, {10, -18, -14, 3, 6, 24, -9, 23, 10, 25, -8, -1, -2, -23, -14, -9}
, {17, -19, 26, 17, 25, -1, -12, 6, -2, 4, -15, -11, 1, -3, 1, 19}
}
, {{-18, -1, -8, 19, -4, -4, 26, 9, 4, 26, -7, 12, 9, 21, -6, 13}
, {25, -22, -8, -5, 8, -3, 19, 7, 7, -17, 16, 4, 0, 16, -3, 23}
, {25, -15, -14, 2, -10, -13, 10, -10, -24, -2, 16, 4, 18, -20, -21, -6}
}
, {{2, -2, 26, -1, 19, 23, -26, 10, 12, 17, 9, 3, 16, -1, 7, 13}
, {22, 10, -8, 13, 21, 1, -26, -19, -2, 1, 10, -24, -11, 13, 14, 0}
, {-25, 11, 1, 24, -19, 24, 7, 24, 19, 2, -10, 26, 7, -9, 21, -13}
}
, {{0, 23, 1, -10, 27, 14, -13, -11, -15, -21, 18, -19, 0, -5, 2, -3}
, {18, 9, -7, -14, 2, 20, 21, 14, 0, 12, -21, 7, -2, -15, 9, 2}
, {-16, 8, -17, -10, 14, 27, 27, -24, 16, 4, -13, 0, 13, 6, -9, -7}
}
, {{13, 1, 19, -6, -24, 11, 15, -18, -8, -20, 26, 1, -14, 20, -5, 13}
, {-7, -14, -11, 9, 12, -2, 11, -8, -19, -16, 20, 10, 3, -6, 21, -20}
, {2, 7, 24, 0, -15, -10, -19, 24, 18, -1, 11, -25, 14, -13, 13, 10}
}
, {{3, 10, 0, -21, 18, -5, -9, 24, -1, 8, -17, 22, -11, -6, 8, -11}
, {21, -10, 19, 1, -22, 5, 4, 5, 1, 21, -7, -12, 10, 6, 4, -16}
, {20, 24, -20, -24, -13, -11, -11, 6, 18, -14, 1, -17, -17, -9, 5, 19}
}
, {{9, -6, 4, -18, 8, 17, 3, -14, -9, 2, 26, 18, 20, 3, 16, -16}
, {-3, -4, 5, 18, 20, 22, 13, -19, 22, 14, -15, -21, 17, -9, 0, 22}
, {20, 10, 24, 0, -25, -11, -7, -17, -19, 21, -25, 2, -7, -4, 12, -12}
}
, {{-26, -25, -12, -5, 27, -2, 19, 19, -8, 9, -21, -27, 22, 5, -21, -1}
, {15, -11, 5, 25, -9, 17, -23, -14, 22, -24, -23, -12, -16, 13, -16, -6}
, {9, -25, 4, -18, 22, 20, -24, -14, 0, 21, -6, 21, -21, -9, 20, 14}
}
, {{4, 4, 6, -9, 7, -11, -21, 11, -8, 11, -15, 27, -23, 6, 17, 18}
, {18, 10, -27, -6, 18, -19, -8, 4, -16, 17, -21, 21, 12, -9, 2, 15}
, {3, 3, 16, 19, 2, 10, 9, 5, -3, -22, -16, 19, 0, -22, 4, -21}
}
, {{-13, -13, -11, -19, 8, 24, -28, 14, -16, 13, 0, -26, 19, 27, 27, -9}
, {-2, -11, 2, -5, -13, -3, 25, 26, -26, -23, 15, 16, -22, 23, -8, -21}
, {25, 9, 13, -10, -9, -18, 12, 14, -7, 12, -21, 19, -23, 6, 23, -18}
}
, {{-15, -8, -8, -1, -25, 8, -26, 13, -19, -23, -13, -14, -26, -28, -14, -17}
, {20, -14, -5, -24, -15, 14, 0, 10, 4, -4, -3, 7, 6, 20, -12, -10}
, {-23, 24, -20, 7, -1, -15, -15, 22, -28, 2, -17, 8, 17, 12, 18, 13}
}
, {{-13, 16, 16, -6, 0, -15, 15, -8, 19, -18, -20, -17, -8, 0, 2, -23}
, {-26, -26, -5, -5, 4, -2, 10, 12, -6, -25, -19, -23, 13, 12, -15, 9}
, {-13, -14, 18, 7, 16, -23, -4, -29, 3, 21, -1, 6, 25, -16, 23, -24}
}
, {{-4, 15, 10, -32, 10, 2, 23, -2, -3, -4, -9, -25, -2, -16, 16, -15}
, {19, 1, 17, 0, 21, 0, -16, 22, 14, -6, 18, 10, -7, -5, -11, 22}
, {8, -15, 9, -13, 20, 20, -20, -5, -9, -29, 20, -18, 1, 6, -13, -26}
}
, {{14, -5, 6, 26, 8, 10, -23, -4, -1, 6, -24, -24, -13, 0, 2, 6}
, {-5, -18, -18, 16, -21, -8, 27, -25, 1, 14, -4, -3, -5, -20, 25, 21}
, {13, 11, 21, 8, -9, 2, 9, -27, -1, -23, -10, 25, -9, -13, 11, -13}
}
, {{-14, -21, -19, -8, -26, -2, -3, -5, 15, 14, 27, 11, 6, -22, 16, -10}
, {7, -22, 7, -21, 0, 0, -23, 0, 15, -25, -26, 9, 23, 9, 19, -26}
, {17, 4, 24, -2, -18, -5, 25, -6, -17, -18, -21, 3, 4, -19, 14, 24}
}
, {{-4, 18, -3, -16, -23, 19, 19, -6, 22, -25, -9, 18, 24, -5, 15, 14}
, {2, -26, 24, 10, -20, -12, 20, 1, 9, 0, -13, -9, 4, -3, 7, -4}
, {-5, 17, 6, -17, 9, 15, -8, 9, -26, 5, 29, -16, -6, 14, -13, 13}
}
, {{-1, -3, 23, 4, 5, 21, 18, -4, 1, -10, 0, 18, 19, -9, -20, -19}
, {-15, -10, 7, -17, 21, -2, -22, -23, 10, -25, -20, -1, -13, -23, 11, -4}
, {-12, 10, 3, 1, 21, 19, 10, -7, -13, 7, 4, -7, -12, 10, -13, -21}
}
, {{17, 1, 22, -21, 8, -21, 12, 21, 26, -20, -13, -18, -5, -4, 24, 6}
, {12, 10, -13, -3, 7, -18, 19, -24, -14, -24, 19, 8, -24, -12, 0, 24}
, {-13, 20, -15, -8, 26, -1, 6, 10, 18, 2, 17, -4, -11, -6, 6, 23}
}
, {{-3, -25, -22, -15, 27, -11, 17, 6, -12, 16, -5, -16, -7, -3, -27, 23}
, {17, 1, 12, -14, -20, -16, 4, 22, 6, -22, 15, 21, -17, -8, -15, 6}
, {-16, -6, 11, 1, -3, -27, -26, 1, 13, -1, -22, -9, -22, -8, 23, 15}
}
, {{25, -3, 3, 26, 5, 3, -21, -8, 18, 7, -23, -9, 26, 1, 20, -2}
, {-13, 9, 8, -20, -18, -19, -5, 21, -25, -11, 4, 1, 3, -23, 13, 8}
, {-24, 21, 9, 5, 3, -13, 22, -18, -16, 25, 21, -9, 16, 24, 17, 9}
}
, {{-25, -12, -23, 17, -13, 11, 2, -7, 16, 0, -14, -10, -3, -9, -2, -3}
, {-13, -8, 24, 2, 16, -24, 17, 9, 26, -7, -22, -16, 22, -1, 18, 8}
, {-27, 24, -24, 11, -25, -11, 1, -21, 1, -28, 20, 11, -6, 6, -18, 11}
}
, {{12, 4, 2, -25, 10, -25, 11, 16, 23, 14, 12, -13, 29, -21, -5, -19}
, {22, 17, 25, 4, -8, -13, 2, -2, 7, 13, 12, -7, 27, -6, -3, 21}
, {-19, -13, 8, 21, -1, -3, 29, -12, -5, 0, 13, 21, 19, -8, 12, 14}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_264_H_
#define _BATCH_NORMALIZATION_264_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       119

typedef int16_t batch_normalization_264_output_type[119][32];

#if 0
void batch_normalization_264(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_264_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_264_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_264.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       119
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_264(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_264_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_264_bias[32] = {99, 33, -78, 3, 26, 115, -53, -111, 154, -11, -24, -73, -185, -51, -57, -9, -96, 43, -23, -7, 119, 48, -25, 21, 43, -92, 38, -75, 86, -92, 46, -182}
;
const int16_t batch_normalization_264_kernel[32] = {272, 248, 247, 350, 261, 293, 300, 289, 333, 212, 260, 254, 289, 269, 286, 240, 299, 281, 206, 200, 249, 191, 253, 287, 291, 384, 214, 307, 272, 318, 211, 308}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_340_H_
#define _CONV1D_340_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       119
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_340_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_340(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_340_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_340.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       119
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_340(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    32
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_340_bias[CONV_FILTERS] = {-1, -1, -1, 0, -1, 0, 0, -1, -1, -1, -1, 0, -2, -1, 0, -1}
;

const int16_t  conv1d_340_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-12, -9, -13, 6, 15, 9, 8, -9, -19, -1, -9, -2, 12, -14, 5, 17, 5, -19, -10, -8, 6, -6, -21, 13, -15, -9, 5, 25, -15, 8, 9, -20}
, {-12, 13, 15, 8, 5, 7, -7, 0, -18, -13, 20, -18, -1, -7, -5, -11, -16, 12, 15, 18, 6, 14, 0, -19, 12, 24, 7, -8, -25, -20, -18, -15}
, {-15, -17, 16, -27, 26, 3, -18, 4, -2, 8, -8, -13, 21, -13, -23, -15, -3, 8, -23, 4, -28, 27, 0, 7, 22, -18, 20, -22, 26, -6, 21, 14}
}
, {{-1, -12, -26, 25, -23, -6, 16, 4, -3, 13, 1, 21, -10, -3, -22, 5, 4, 16, 19, -5, 11, -14, -28, 6, 24, 3, -17, 13, 15, -25, 20, 7}
, {-21, 10, 3, 11, 3, -19, 7, -16, -9, -12, -8, -18, -9, 21, -19, -13, -9, -22, -26, -23, 4, 6, 9, 18, -5, 19, -25, 21, -6, 5, 5, 21}
, {9, -20, -15, 20, 20, 24, -12, 11, 25, 10, 24, -14, -8, -12, -14, -13, -29, -14, -5, 5, -7, 3, 6, -19, -6, 3, -17, 1, -10, -7, 19, -29}
}
, {{-14, 22, -24, -1, -12, 26, -28, -21, -22, -3, 23, 7, 15, -16, -21, -20, 16, -29, -23, 14, 18, -9, 15, 17, -10, -20, -8, -8, -24, -22, -27, 25}
, {18, -14, -11, 14, -4, -12, -20, -19, 15, -15, -21, -14, -18, -20, 1, 4, 11, -14, -25, 19, 0, -23, -3, -20, 3, 22, -29, 20, -20, 14, -7, -6}
, {-18, 18, 7, -11, 20, 16, 8, -2, 11, 13, 1, 2, -6, -8, 20, 5, 1, 26, 1, -2, 22, 6, 22, -20, -15, 2, 22, 13, 15, 7, -25, -8}
}
, {{-5, -1, -4, 17, 1, 21, 8, -23, 2, -10, 18, -25, -18, 20, 12, 9, -23, 24, 13, -19, -20, -9, 20, -21, -18, -16, -12, 5, 8, 1, 26, -9}
, {1, -4, -5, -14, -13, 27, 16, 16, 11, -11, -3, 10, -19, 13, 20, 18, 14, 1, 17, 13, 14, 15, -18, 6, -11, -5, -1, 12, -2, -16, 3, 1}
, {22, 21, 16, -27, 5, 15, 23, -13, 15, 29, 29, 9, 11, 1, -8, 0, 6, -13, -6, -19, 20, -18, -19, 16, -1, -4, 1, 14, -23, 0, -18, -22}
}
, {{-4, 9, 2, -12, -4, -8, -23, -1, -22, -18, 13, 8, -4, -9, 19, -10, -24, -9, -12, 18, 9, 1, -6, -9, 13, -2, 18, 7, 5, 0, 3, -13}
, {-25, -10, -18, -10, -1, -7, -13, 9, -6, 19, -22, 4, -1, 24, -17, 17, 24, 4, 23, -1, 16, 23, 2, -2, 28, -25, 8, 0, -18, -5, -16, 10}
, {-12, -11, 24, -13, -18, -28, 16, 16, 8, 19, 9, 2, 23, 8, -15, -15, 24, -21, -10, 0, -22, -16, -23, 2, 17, 5, 8, 1, 21, 16, 23, 14}
}
, {{-1, -11, 10, 15, -6, 2, 3, 14, -25, -2, 21, -27, 0, -17, 8, -27, 14, -12, -26, -22, -17, 9, 21, 11, -13, 2, -15, -26, -19, -7, -4, 15}
, {-14, 21, -19, 8, -11, -21, 21, 4, 23, -21, -8, 8, -6, -22, 9, 15, -20, 0, -27, 0, -11, -7, -10, -5, -20, 8, -24, -1, 8, 27, -23, 26}
, {-12, -21, -22, 12, 13, -16, -24, 2, -27, -11, -25, 22, 7, -3, -8, -5, -20, -23, 21, -25, -18, 16, 8, -25, -18, 20, -17, 15, -18, -26, -24, 25}
}
, {{-7, -11, 11, 10, -2, -15, 0, 8, 3, -4, -4, -8, 26, 11, -1, 4, 1, -28, -7, 1, 16, -22, -22, -26, 5, -24, -26, -4, 23, 23, -27, 9}
, {-10, 7, 23, -10, 4, 19, -15, -10, -20, -22, -13, 10, -9, -15, 7, 18, -16, -6, -23, 19, -7, 3, -5, -8, 14, -7, 3, -1, -7, 22, 0, -13}
, {-7, 17, -1, 26, -14, -22, 24, -25, -5, -2, 25, 13, -17, 7, 22, -21, 13, 25, 19, 2, -23, -15, 18, 3, 28, -5, -9, -1, -7, 25, -27, 22}
}
, {{3, -5, 4, 11, 22, 23, 12, 11, 2, -7, 16, -25, 16, -15, 15, 3, -17, -1, 3, -2, -11, 23, -5, 21, -16, 13, -25, 16, 3, 9, 9, 19}
, {25, 2, 10, -27, 0, 25, 23, -21, 1, 27, 15, 10, -11, -23, 5, 21, -16, -1, -20, -17, 8, -10, -16, 25, -20, -13, 19, -9, 14, 1, -14, 0}
, {12, 16, 13, 17, 7, 23, 9, -6, -26, -12, -15, 6, 3, 13, 15, -24, 2, -2, -16, 3, 21, 2, 28, 20, 9, -2, -16, -19, 16, -15, -9, 18}
}
, {{-5, 14, -15, 14, -8, 8, -11, -23, -7, -5, -3, -3, 5, -19, -16, -1, 23, 5, 20, -23, -4, 14, -23, -6, -9, 16, 24, -17, 5, 17, 5, 17}
, {5, 16, -18, -26, -25, -19, 11, -16, -8, 0, 2, 4, -13, -24, 20, 13, -15, 18, 22, 21, 5, -13, 3, -3, 13, 20, -21, -2, -5, -12, -23, -4}
, {10, 5, -21, 11, 23, -22, 18, -4, -20, 23, 7, -15, 22, -1, -1, 5, -3, 16, -27, 23, 21, -26, 15, -14, -17, 22, -4, -21, -17, 24, 0, -16}
}
, {{-20, 24, 22, -29, 24, -2, 14, -5, -15, 12, -18, -14, -24, 27, 6, 5, 19, 28, -12, 16, -6, -19, -8, -28, -11, 6, 12, 18, -14, -16, 25, 13}
, {17, -22, -16, -31, 0, -6, -20, -19, 5, 14, 15, -26, -2, 28, 10, 7, -15, 25, -20, -13, 10, 17, -11, 17, -23, 5, 0, -16, -26, -1, -9, -15}
, {0, -2, -8, -9, -17, -5, -4, 2, -15, 6, 16, -1, 7, 7, -17, -27, 10, -10, -16, 3, 16, -18, 8, -12, -20, 16, 22, -8, 6, 22, -14, -15}
}
, {{-8, -19, -4, -7, -8, 5, -10, 21, -28, 7, 15, 19, -22, -21, 16, -4, 9, 17, 21, -9, -21, -21, 17, -17, -22, -18, -22, -12, 11, 6, 16, 16}
, {-4, -14, -7, 16, 7, -18, -21, -18, 6, -11, 22, -12, 26, -9, -22, 15, 23, -24, 3, -12, 1, 0, -26, -5, -3, 16, 16, -2, 14, 19, -12, -3}
, {-8, 17, -17, 12, 21, -27, 24, -28, 15, -18, 18, 10, 24, -19, -16, 17, 2, 3, -21, -3, 18, 6, -25, 1, 1, -4, 7, 15, -14, 6, -14, 20}
}
, {{25, -13, -11, -26, -9, -27, 2, 18, 14, 3, -18, 15, -20, 23, -17, 8, -7, 1, 3, -24, 0, 20, -17, -20, -9, -9, 11, 10, 2, 19, -14, -22}
, {-16, -25, 5, 0, 11, 14, 22, -12, 18, -20, -19, -22, -1, 11, -19, -4, 5, -21, -9, 21, 24, 14, 3, 1, 2, 9, 6, -20, -6, -1, 24, 25}
, {-2, -18, 18, 16, -8, -13, -11, -4, 26, 21, -19, -2, -17, -18, -17, -22, -23, 11, 2, -5, 22, -3, 4, 21, 15, -17, 14, 7, 20, -19, 3, 0}
}
, {{18, -10, -21, -26, 14, 16, -3, 3, -17, -24, -21, -24, -14, -27, 21, -17, -6, 26, -21, -18, 14, 11, 9, -3, 21, -15, 9, -17, 0, 1, -4, 7}
, {-24, 18, -13, -25, 10, 3, -8, 24, 1, 6, 20, 24, -15, 4, -14, 11, 6, 21, -9, -20, 18, -1, -9, 22, -2, -14, 0, -17, -3, 6, -24, -13}
, {-22, 26, -19, -4, 17, 16, -23, -20, 6, -8, -20, 6, 20, 4, 3, -24, -11, -29, -26, -18, 5, 10, 1, -22, 15, 4, 23, 18, -12, -11, -26, 21}
}
, {{13, 7, 4, -9, 15, 23, 12, 1, -7, -5, 18, -11, 18, -16, -18, -9, 17, -15, -7, 11, 13, -6, -18, -20, 21, -24, -14, 0, -15, -7, -28, -6}
, {20, 16, 10, -17, -23, -14, -10, 3, 14, 13, 1, 11, 1, -5, -22, -19, -30, 4, -12, 21, -22, -30, 1, -26, 4, 26, 11, 4, -4, -19, 7, 15}
, {-25, -15, -9, -5, -5, 24, -6, 16, 13, -22, -3, 15, 0, 25, -14, -19, -12, 27, -9, -24, -21, -23, 20, 11, -24, 21, -28, -15, 8, -4, -23, 1}
}
, {{11, -24, 6, 18, -27, -13, 3, -9, -12, -9, -19, 0, -20, -25, 1, 17, 17, 2, 12, 2, -11, -14, -19, -8, 9, 25, -16, 4, -24, -1, 2, 12}
, {20, -12, -4, -13, 8, 15, -29, 7, 16, -4, -8, 14, 18, -1, -19, -2, -6, -16, -12, -4, 2, -17, 19, -26, -5, -9, -8, 7, -17, -6, -3, -19}
, {-3, -17, 18, 6, 13, -23, -11, 6, 19, -7, -26, 12, -13, 19, -20, 3, -5, 21, -22, -26, -13, -27, 25, -10, 14, -5, -9, 26, 17, 7, -17, -22}
}
, {{26, -5, -25, 13, -1, 3, -13, 6, 21, -13, 1, -10, 16, -6, -8, -19, -2, -16, -6, 13, -9, 1, 2, -13, -14, -6, -15, 15, -22, 3, -22, 10}
, {8, -16, 21, 5, 14, -17, -4, 14, -18, -24, 8, 11, 20, 20, -6, 21, -15, -4, 7, 6, 7, 15, 4, -3, 13, 9, -5, 29, -24, 12, -15, 26}
, {1, -19, 1, -2, 24, -1, 12, -2, -3, 10, -2, -22, -11, 22, 23, 7, -4, -13, 15, 22, -26, -22, -10, -3, 8, 12, 20, -15, -15, -9, 6, -8}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_265_H_
#define _BATCH_NORMALIZATION_265_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       117

typedef int16_t batch_normalization_265_output_type[117][16];

#if 0
void batch_normalization_265(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_265_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_265_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_265.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       117
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_265(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_265_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_265_bias[16] = {41, 49, 79, -73, -36, 148, 8, -114, 8, 57, 30, 2, 80, 113, 131, -23}
;
const int16_t batch_normalization_265_kernel[16] = {191, 210, 220, 175, 221, 137, 209, 150, 184, 175, 188, 178, 153, 213, 193, 178}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_341_H_
#define _CONV1D_341_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       117
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_341_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_341(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_341_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_341.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       117
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_341(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_341_bias[CONV_FILTERS] = {-1, -1, -1, -1, 0, -1, 0, 0, -1, -1, -1, 0, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, -1, 0, -1, -1, -1, -1, 0}
;

const int16_t  conv1d_341_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-1, 12, -5, 23, 25, 5, -20, 16, -13, -3, -9, 4, -15, 18, -6, -6}
, {8, 8, -4, -3, 9, -8, -13, -12, -8, -18, 10, 24, 14, -6, -21, -23}
, {-11, -2, 14, 10, -20, -26, 21, -6, 20, 12, 9, 20, 26, -20, -25, 17}
}
, {{-12, -16, 11, 19, -23, 2, -19, -7, 2, -16, 3, 14, -27, 23, 16, 12}
, {8, 0, -7, 16, 7, 20, 0, -27, -11, -17, 9, 7, -18, 1, -10, -6}
, {11, -11, -8, 20, 10, 17, -12, 20, 13, 10, 15, 29, -15, 13, -24, -6}
}
, {{-23, 7, 19, -10, -7, 14, -25, 22, -21, 2, -10, -3, -18, -11, -20, -7}
, {-29, 9, -9, -18, 16, 1, -22, -2, 25, -16, -15, -20, -6, 2, -16, 22}
, {-17, 6, 27, -3, -22, -9, 6, 8, -16, -10, 20, 24, 5, 20, 1, -20}
}
, {{-24, -23, 20, 23, 8, 18, -7, -19, -22, -8, 19, -6, 7, -2, 22, 21}
, {-20, 9, -8, 11, 22, -27, -11, -4, 4, 22, 1, -20, 25, -10, 14, -4}
, {-2, 15, -18, -14, 23, -1, 15, -20, -25, -12, 15, 24, 6, -24, 14, 22}
}
, {{2, 22, 11, -22, -9, -4, 19, 16, 3, -9, 3, -24, -13, -16, 18, -24}
, {-1, -26, -14, -9, 22, -3, 5, -2, -3, -4, -14, 25, -13, 1, 13, 24}
, {-25, -16, -25, -22, -5, 6, 15, 12, -13, -2, -14, 14, 18, 20, 3, -10}
}
, {{16, -22, -17, -20, 12, -16, 2, -20, -13, -8, 14, -11, -9, -19, -17, 20}
, {12, -23, -24, 6, 22, -11, 3, -25, -17, -20, -2, -27, -7, 11, -29, -22}
, {30, -16, 9, 11, 23, 16, 20, 15, 14, -19, -5, -19, 2, 24, 5, 11}
}
, {{-16, 2, 2, 10, 1, -6, -2, 2, -16, -21, 15, 14, -9, -20, -18, 21}
, {19, -18, -17, -4, -1, -9, -19, -18, 19, 18, 8, 7, -23, 14, -23, -4}
, {16, 3, -6, 6, 15, 11, 3, -12, 9, -14, 10, 27, -28, -6, 25, 13}
}
, {{-20, 4, 5, 14, 18, -3, -15, -18, -17, 11, 16, 19, -7, 14, 24, 11}
, {14, 19, 9, 3, 4, 22, -13, 0, -19, -4, -25, -20, -23, 8, -21, -1}
, {-23, -13, 17, -22, 15, -20, -22, 19, -18, -24, 1, 0, -30, -17, 22, 21}
}
, {{-9, -6, 4, -17, -24, 6, -14, 15, -2, 21, 22, -3, 19, 24, 14, -10}
, {6, -10, -22, -13, -27, -17, -24, -26, -22, -5, -5, 9, -22, 20, 4, -3}
, {14, 4, 21, -14, 19, -24, 10, 14, 11, 2, 8, 18, -19, 21, -23, -7}
}
, {{-12, -7, 22, -5, 24, 10, -27, -27, -1, -1, -19, -22, -20, -17, 2, 10}
, {29, 19, -13, 11, 19, -25, 18, 4, -20, -5, -7, -25, -17, -21, -18, 2}
, {-7, 3, -5, -15, -20, 17, 9, -26, 22, 20, 25, -26, 15, 5, 0, -3}
}
, {{-11, -23, -6, -10, 1, -21, -4, 0, -13, -23, 10, -17, -17, -4, -6, 25}
, {-17, 3, -23, 27, 24, -11, 17, 25, 17, 25, -25, -10, 4, 12, -10, -24}
, {8, -12, 16, 23, -7, -8, 24, 0, -20, 24, -6, 4, 23, -7, -17, -2}
}
, {{-13, -12, 3, 24, 23, -22, -12, -13, -1, -2, -6, 16, 21, 4, -16, 3}
, {-14, -2, -19, 7, 18, -16, 23, -22, -14, -17, 4, -1, 21, 23, 15, 22}
, {21, 26, 16, -6, 6, 1, 19, 4, 11, -2, 7, 8, -1, -3, -11, -6}
}
, {{-2, 3, -23, 17, -23, -10, -13, -20, -24, -5, 12, -1, -24, 27, -20, 6}
, {3, 23, -21, -3, -6, -20, -3, 14, -3, -1, -15, 3, 5, -14, 2, 17}
, {0, 19, 0, 15, -21, 16, 23, -3, -21, 20, -15, 11, 4, 14, 24, 0}
}
, {{-7, -4, -21, -2, 25, -17, 21, 10, -9, 25, 20, 20, 24, 20, 25, 16}
, {8, -1, -23, -10, 10, -14, 8, -1, 9, 4, -22, 14, 14, 12, -24, -17}
, {14, -2, -14, 12, -22, 8, -28, 12, 0, 18, -14, 3, -21, -10, 6, 2}
}
, {{-14, 6, -18, -6, -25, -22, -10, 22, 6, 2, 27, 0, -13, 3, -13, -19}
, {20, 7, -26, -23, -2, -18, 15, 7, 11, -15, -9, -26, 20, 9, 24, 19}
, {23, -17, -22, -3, 23, 16, 0, -4, 12, -22, -1, 10, -16, -21, -4, 16}
}
, {{0, 19, -13, 3, 3, -7, -5, -7, 23, -3, 4, 8, 15, 7, -17, -28}
, {-22, -25, 7, 6, -3, 2, 16, 7, -22, 23, -11, 6, -10, -23, 18, -8}
, {28, -24, -16, -6, -8, -16, -16, 14, 21, -4, 12, 23, -11, -2, -9, 17}
}
, {{8, 8, -2, -25, -20, 12, -28, -16, 22, 20, -14, 0, -14, 21, 3, 18}
, {-15, 7, -17, 22, 3, 6, 21, 18, 18, 14, -22, 21, 16, 0, -18, 6}
, {-10, -23, -17, -23, 11, 20, 4, 17, 15, -24, 10, 1, -9, -19, -5, -3}
}
, {{-9, 20, -21, -11, 4, 8, 22, 17, -25, -1, -6, 11, 0, 3, -17, -11}
, {-25, -2, 19, 1, -17, -19, 11, -7, 27, 13, -18, -9, 13, -14, -21, -23}
, {-23, -22, -21, 22, 16, 1, -10, 12, 23, -7, -5, 4, -12, 14, -19, 27}
}
, {{6, -26, -11, 18, 1, 10, 20, -26, -26, -10, 17, 18, -21, 21, 4, -24}
, {24, -9, 1, -4, 24, -7, 18, 15, 7, -6, -7, -3, -7, 9, -10, 6}
, {-25, 0, -28, -22, 13, -22, -18, 11, -23, 6, -6, 2, -15, 4, 14, 8}
}
, {{0, -5, 10, -15, 22, -20, 10, -1, -19, -11, -29, 18, -15, -13, 10, 20}
, {28, -24, 2, -27, 15, -7, 18, -26, 6, -14, -19, -22, -4, -27, 2, 11}
, {-4, 17, -16, -18, 0, -7, 10, 23, 15, 24, 4, -4, -13, -12, -6, 8}
}
, {{4, 18, -20, 5, -23, 14, 0, 12, -4, 14, -17, -23, -1, 27, -9, 13}
, {-13, -11, -6, -19, 18, -1, 7, -7, -1, -1, 14, 11, 20, -9, 25, 26}
, {-19, -24, 22, 11, 20, 9, 23, 3, 26, -5, -25, -18, 5, 0, -4, 22}
}
, {{26, 23, 15, 15, -19, -26, 21, -10, -11, 4, -12, 3, -3, -6, -8, -22}
, {-1, -24, 3, 28, 18, -2, -15, -24, -22, -22, 5, -2, -27, -2, 13, -14}
, {-10, 20, 18, 18, -12, -5, -25, -23, 19, 6, 4, -6, 0, -23, -11, -24}
}
, {{-11, 0, -16, 12, -2, 9, -17, 24, 12, 18, 22, -22, 25, -4, 16, 19}
, {11, 8, -2, -20, 10, 20, -9, 2, 11, -19, -16, 26, -31, 21, -25, 11}
, {6, 23, 5, 15, 25, 19, -13, 23, -27, 8, 13, 8, -10, 15, 10, -13}
}
, {{-30, -21, -26, 2, -18, -7, 4, 1, -17, -23, 1, 11, -6, -14, 19, -9}
, {26, 13, -7, 22, -22, -8, 7, -7, -23, -8, 3, -11, -5, -4, 2, -23}
, {7, 21, -5, -23, -3, 25, 18, -12, 5, 15, 11, 19, -8, 21, -26, -20}
}
, {{-3, 14, -16, 11, 1, 13, 19, -1, 22, 9, -13, -21, -10, 1, -17, 11}
, {-7, 18, 8, -1, 4, -3, -32, -11, -10, 20, 0, -1, 4, -12, 14, 21}
, {-8, -20, 13, -17, -24, 21, -25, 3, 16, 7, -14, 14, 24, 11, -20, -19}
}
, {{-19, 17, -16, -1, 9, -13, -16, -22, 8, 5, 8, -24, -6, -14, -3, 1}
, {-26, -6, 28, 9, -20, 26, -20, 13, 18, 9, -23, 11, -22, 8, 24, 23}
, {5, -16, 17, -19, -19, -26, 20, -10, -13, 8, -8, -3, 1, 11, 20, 22}
}
, {{-14, -4, -15, -5, 6, -2, -19, 5, -20, -1, 10, 0, 12, 3, -2, -9}
, {-14, 1, -9, 17, 15, 19, -3, 10, 9, 21, 0, 10, -1, -10, -16, 22}
, {-9, -8, 26, 4, 21, -22, 11, 17, 27, -16, -16, 22, 16, 19, 2, -10}
}
, {{-14, -5, -15, -19, -20, -12, 14, 1, 20, -27, 14, 11, -2, -20, -5, -10}
, {-1, 18, -25, 25, -6, -23, 29, -22, 15, -22, -8, -5, 4, 19, -17, -16}
, {-11, -10, 23, -9, -20, 5, 8, -25, -13, -7, 11, -17, -12, -26, 6, -14}
}
, {{-17, 8, 26, -14, 10, 22, -15, 21, -7, 20, 1, 0, 19, 8, 24, 1}
, {19, 6, -2, 20, -9, 8, -1, -20, 21, -16, 18, -5, -6, -20, -21, -20}
, {-6, 0, -24, 15, -22, 12, -23, 15, -20, -17, -4, -15, -17, 4, 22, -11}
}
, {{-24, 17, -14, -9, -31, -12, -20, 4, -7, -17, 11, 22, -26, -4, 1, -13}
, {1, -9, -2, 19, 14, 19, -5, -28, -1, -15, -21, 23, 15, 3, -16, 1}
, {-24, -23, -22, 10, 6, -10, -13, -4, -17, -21, -17, -8, -15, 20, -10, 17}
}
, {{11, 6, 2, -18, 4, -28, 0, 7, -23, -1, 10, 6, -28, 1, 26, -18}
, {-19, 12, 20, -23, -15, 3, 4, 14, -19, 10, -2, -11, -26, -8, -26, -7}
, {18, -29, -12, 19, 19, -9, -11, -25, -22, -7, -12, -22, 23, -8, -19, -13}
}
, {{-26, -13, -16, -6, 27, 14, -9, 8, 18, 18, 7, 1, -15, 7, 21, 13}
, {-11, -11, 10, -9, 11, 3, -2, -7, -19, -23, 20, 15, 27, -2, 19, -11}
, {-25, 2, 5, -19, 15, 21, -7, -9, -6, -18, -5, 7, -12, -17, -3, 0}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_266_H_
#define _BATCH_NORMALIZATION_266_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       115

typedef int16_t batch_normalization_266_output_type[115][32];

#if 0
void batch_normalization_266(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_266_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_266_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_266.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       115
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_266(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_266_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_266_bias[32] = {-29, -37, 75, -38, 30, 45, 2, 44, 27, 53, 7, -81, 17, -65, 38, 4, -22, 35, 27, 45, -70, 63, -131, 75, -9, 20, -103, 169, -5, 136, 106, -3}
;
const int16_t batch_normalization_266_kernel[32] = {269, 240, 251, 214, 241, 196, 257, 282, 242, 235, 230, 231, 262, 259, 271, 291, 255, 247, 251, 214, 204, 212, 221, 239, 277, 244, 265, 273, 264, 235, 194, 295}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_342_H_
#define _CONV1D_342_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       115
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_342_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_342(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_342_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_342.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       115
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_342(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    32
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_342_bias[CONV_FILTERS] = {-1, 0, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0, -1, 0, 0, 0}
;

const int16_t  conv1d_342_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-9, 20, 24, -8, 8, -2, 7, 13, -1, -18, 5, 1, 7, -12, -13, -6, 19, 7, -14, 6, -15, 16, -6, -17, 18, 3, 17, 13, 24, 22, -25, 25}
, {-8, 21, 24, -1, -4, 21, -15, 14, -6, 13, 7, -5, -24, 1, 18, -13, -9, -18, -2, -14, -22, 17, -11, 15, -16, 27, -5, -22, -1, 11, 3, 9}
, {-4, -15, 21, 18, -5, 2, 28, -13, 11, -14, 22, 1, -15, -1, 0, 1, -8, 13, 25, 13, -12, 12, 23, -11, 9, -20, -14, 2, -27, 10, -7, 23}
}
, {{-20, -21, 1, -3, 13, -14, -11, 0, 11, 18, 24, -19, 14, 9, -1, 25, -15, 18, -18, 6, -7, 18, -23, -24, -7, -1, 5, -1, -16, -29, 27, -19}
, {-17, -2, -20, -17, -2, -13, 21, 21, 23, -6, 6, -6, -9, 2, -11, -9, 6, -15, -5, 20, 21, -11, -21, -19, 23, -16, 5, -20, 19, 14, 11, 6}
, {24, 9, -22, -3, -2, -8, -9, -16, -8, -19, 22, 7, -17, 9, 6, 24, -22, -3, -23, 20, 23, 12, 11, -14, -1, -20, -12, 3, 16, 20, 14, -6}
}
, {{-11, 6, 21, 20, -22, 19, 10, 9, -19, -9, -22, 23, -24, -27, -1, -19, 18, -20, 19, 2, 1, -9, 15, 17, -11, -25, 16, -21, 14, -12, -2, -3}
, {-23, 21, 18, 11, -17, 21, 21, 18, -19, -19, -14, 22, -21, 8, -21, -23, 14, -25, -6, -19, -7, -24, -1, -15, -20, -13, -18, -27, 18, 24, 13, 15}
, {0, -20, -2, -1, -5, -1, 20, -26, -23, -13, 17, 23, 2, -4, 16, 12, -6, -16, -24, 28, 21, 12, -3, -19, 5, -6, 26, 21, -27, -10, -25, -24}
}
, {{-17, 10, -22, 20, 16, -15, -12, -24, -4, -5, -7, 0, 8, 7, -4, 4, -8, -2, 6, 20, -8, -16, -9, 4, -20, 9, -2, -7, 15, 21, -8, 20}
, {6, -15, 6, -19, 25, -8, -17, -15, -15, 21, 11, 7, -20, -5, 19, -29, -10, 16, -16, -1, -9, -6, 3, -7, 8, -22, -4, 4, 19, -2, 23, 9}
, {6, -19, -20, -13, -27, -6, 17, 16, -2, -17, -20, -15, 19, 13, -22, 1, 12, -3, -8, -7, -19, -18, -15, -8, 19, -11, -15, 0, 15, -15, 17, 23}
}
, {{-7, 13, 9, 23, 16, 7, 19, -18, 5, 20, 20, 26, -15, 18, -5, -23, 13, 20, 24, 24, -10, 8, 15, 13, -27, -12, -10, 13, 28, 16, 22, 5}
, {-18, -10, -26, -8, 20, 17, -21, -11, -26, 3, 21, 6, -12, -29, -16, 22, 23, -18, 5, 23, 23, -8, -24, -28, -14, -23, -15, 5, -13, -5, 10, -12}
, {-22, 4, -22, -21, -26, 10, -26, -12, -15, 17, 22, -7, 8, -13, -4, -5, 23, 9, -14, -21, -17, 10, -6, 21, 28, -4, 14, -8, -17, 5, -16, 23}
}
, {{-27, -23, 16, -14, 13, 11, -18, -22, 0, 10, -19, 3, 4, -20, -11, -9, 11, 23, -18, 23, 24, -1, -7, -22, 4, -9, 4, -29, -21, -5, 21, 11}
, {-8, 19, 24, 8, -12, -10, 23, -1, 3, 9, 5, 17, 9, -3, 18, 24, 2, -23, 14, -14, 6, -17, 9, -3, -2, 17, -19, 6, -23, 15, -24, 4}
, {-8, 4, -9, 9, 5, 22, -26, 0, -6, 23, 1, -16, 2, 23, -23, -15, -5, -16, 13, 23, -12, 4, 0, -17, 5, 17, 14, -8, -17, -24, -5, -20}
}
, {{-14, -4, 24, 10, 3, 2, -5, 5, 17, 11, 6, -16, 18, -18, 20, -22, -11, 16, -17, -4, -7, 22, 16, 8, 18, 22, 22, -4, -8, 10, -28, 20}
, {25, 2, 18, -22, 19, -1, -12, -1, -10, 12, -4, 14, 25, -24, -17, 14, -23, -24, -20, -22, -12, -6, 6, 17, -9, 22, -12, -23, 7, -9, 10, -4}
, {-6, -5, 0, 13, 1, 10, -9, 2, 27, -8, 14, 22, 19, -7, 26, 22, -23, -3, -21, 15, 11, -14, 16, -23, -12, -15, 20, -21, 1, 6, 24, -9}
}
, {{-3, 11, -7, 8, 0, -9, -8, 14, 21, -4, -18, -4, 9, 0, -2, 6, -19, 24, -15, -10, 12, -8, 16, -2, -8, -12, -24, -6, -25, 7, 10, 11}
, {-16, 20, -27, -21, -15, -2, -1, 15, -8, -22, -11, -12, -11, -16, -16, 1, -16, 1, 12, 7, 4, -19, -12, 10, -1, 16, 0, -8, 4, 4, -9, -24}
, {-3, 15, -10, 17, 18, -14, 4, 2, -2, 9, 4, 4, 3, -7, -8, -15, -27, -10, 0, 10, 21, 11, -6, 12, -3, 8, 11, 20, -27, -2, 22, 16}
}
, {{21, 14, -21, -24, 12, 1, 3, 22, -19, -1, -4, 19, -31, -27, 15, -22, -5, -24, -13, 13, 23, 24, -20, 13, 7, 7, 10, -20, -2, -12, -19, 15}
, {6, 20, -12, -2, -12, -4, 22, 3, -19, -3, 4, -22, -5, -24, -10, 20, 0, -20, -23, -21, -5, 17, 3, 18, 19, -17, -16, 0, 0, 6, 7, 8}
, {10, -21, 10, 4, -26, -8, -9, -15, -24, -22, -21, 11, -2, 24, -1, -15, -3, -10, 22, 1, -28, -1, -25, 4, -22, -25, -25, 2, 0, 5, 24, 8}
}
, {{24, -21, 12, 1, 22, -8, -17, 15, -9, 22, 1, 8, -5, -24, 5, 24, 9, 4, 12, -8, 1, 7, -22, -12, -2, -14, 19, -10, -1, -11, 5, 15}
, {21, -19, 2, 5, -8, 18, -9, -1, 8, -2, -17, -26, 4, -18, -14, -12, -13, 18, -5, -4, -13, -7, 3, -1, 6, 12, 11, -15, -7, 24, -13, -26}
, {21, 1, 13, -3, -9, -1, 18, -19, -6, 7, -19, 8, 21, 18, -24, -16, -26, -23, 13, 0, -8, 24, -2, -13, 11, -8, 13, 12, -17, 2, 14, 17}
}
, {{24, 19, 4, -6, -11, 21, -17, -26, 5, 6, -18, -4, 20, -22, 19, 23, 6, -13, 7, 7, 10, -19, 14, -31, 6, -5, -19, -27, 14, -15, 23, 3}
, {4, 16, 15, -20, 7, -23, -4, -13, -7, 9, 17, -17, -23, -1, -2, -10, -22, -2, -1, 4, -3, 22, 19, 1, -2, -5, 15, -16, 21, -1, -7, -19}
, {-17, -8, 2, 17, 24, 27, -10, -26, -6, -18, 17, -23, 22, -20, -18, -1, -3, -25, 19, 24, 15, 2, -27, -9, -3, -26, 16, 3, -20, -5, 12, 15}
}
, {{-8, 23, -1, -24, 28, -21, -7, 13, -21, 9, 16, 7, 9, 14, 14, 14, -4, -4, -15, -12, 20, -15, -9, 18, 10, 12, 5, -16, 21, -12, -6, -25}
, {-10, -23, 21, 17, -23, 7, -20, -8, -22, 17, 18, -8, 6, -24, -9, 16, -27, 8, 9, -2, 8, -18, -1, -3, -10, 0, -19, 26, 15, 17, 24, -5}
, {-13, -20, 23, -3, -5, -13, -27, -3, 24, 15, 20, 17, -27, 12, -24, -5, -9, 7, 17, 1, 10, 1, -3, -5, -13, 7, -11, -15, -23, 6, 16, 7}
}
, {{25, 13, 13, -1, -21, -12, 19, 15, 10, 11, -20, -12, -16, 0, -11, 19, -17, 24, 0, -13, 2, -10, -27, -7, -12, -13, -5, 26, -10, -12, -20, -8}
, {-20, -7, -1, -21, -1, -4, -6, -12, -17, 6, -8, 9, -13, -12, -25, 18, 4, -15, -5, 22, 12, 27, 16, -13, 17, 1, -2, 15, 18, -16, -10, -15}
, {-25, 1, -24, 15, -22, -14, -3, -12, -3, -22, 26, -26, -2, -11, 20, 3, 16, -21, -18, 1, -16, 11, 9, 3, 24, -27, 22, 3, 20, -6, 20, -11}
}
, {{21, -12, 2, -5, 1, -19, -23, 13, 22, -3, -12, -8, -21, -1, -7, 12, -20, 0, -21, 26, 21, 7, 0, 14, -7, -15, 13, -1, -11, 23, 20, 10}
, {-5, -9, -19, -11, -6, 10, -10, -24, -22, -16, -21, 2, -13, 2, -15, -25, 14, 17, 13, 3, -6, -4, 11, -16, 13, 16, 27, -17, -10, -21, -1, -28}
, {-6, -11, 15, 0, -10, 13, -17, -20, -22, 6, -18, 1, -18, 9, 11, -18, -11, -17, -1, -24, -15, -8, -27, 14, -16, 15, 3, -8, -7, -12, -19, 20}
}
, {{-29, -4, 11, -24, 9, -5, -1, 2, -17, 8, 11, 5, -1, -9, 21, -17, 21, 15, 3, -20, -19, -24, -6, 4, -22, -13, 20, -13, 6, 17, 6, -19}
, {15, -15, -7, 24, -19, 10, -3, 1, -3, -15, 29, 6, 18, -4, -15, 12, 3, -22, 19, -16, 17, 11, 11, 1, -22, -19, -5, -11, -6, -17, -27, -2}
, {-1, -10, -17, -20, 26, -14, -2, 22, 24, 15, -15, -13, -8, 2, 9, -9, -23, 24, 6, 23, -16, -25, 16, -19, -28, -11, 9, 4, -2, -16, -30, 6}
}
, {{-5, 7, -26, -1, 6, 29, 21, -26, -24, 24, -4, 7, -7, -23, 14, -10, 14, -26, 3, 8, 1, -24, -6, 14, 22, -1, -27, 17, 9, -16, -19, 17}
, {18, -22, -23, -24, 24, -6, 17, 16, -10, -15, 20, 15, -20, 17, -20, 3, -21, 25, 14, -2, 26, 24, 15, 6, -22, 21, -2, -1, 18, -25, -6, 22}
, {6, -25, -13, 1, -11, -22, 14, 13, 1, -18, -22, 9, 15, -22, -30, 1, -6, -8, 15, 25, -12, -5, -3, -9, -15, 11, -7, 20, 23, 26, 17, 7}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_267_H_
#define _BATCH_NORMALIZATION_267_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       113

typedef int16_t batch_normalization_267_output_type[113][16];

#if 0
void batch_normalization_267(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_267_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_267_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_267.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       113
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_267(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_267_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_267_bias[16] = {-91, 6, 49, 50, -25, -4, -63, 52, 89, 7, 8, -24, 57, 103, 80, -29}
;
const int16_t batch_normalization_267_kernel[16] = {165, 199, 142, 140, 144, 188, 146, 229, 138, 194, 173, 192, 162, 149, 185, 160}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_343_H_
#define _CONV1D_343_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       113
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_343_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_343(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_343_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_343.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       113
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_343(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_343_bias[CONV_FILTERS] = {-1, -1, 0, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0, 0}
;

const int16_t  conv1d_343_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-24, 11, 18, 14, -4, -7, -16, 3, 15, -23, 7, 10, -11, -6, -15, -22}
, {-20, 17, 21, -26, -25, -16, 13, -31, -7, -16, -5, 18, 11, 12, -1, -26}
, {-5, -7, 1, -10, 18, 20, 10, 21, -2, 7, -25, 3, -9, -22, 3, 20}
}
, {{17, 10, -15, -23, 4, -5, -18, 15, 11, 17, 23, -22, 7, -8, 2, -5}
, {23, 18, 24, 18, 24, -18, -1, 7, 11, 5, -4, -23, -9, 12, -5, 21}
, {-1, -23, 26, -7, -13, -1, -20, 9, 2, -12, 7, -1, 25, -15, -26, -11}
}
, {{-24, 13, -5, -5, -19, -4, -20, 31, -6, 1, 20, 16, -15, 21, -6, 24}
, {24, 4, -27, -7, 24, 2, 8, -24, 13, -7, -3, -1, -13, -17, -19, -19}
, {-22, 0, -11, 4, -18, 1, 26, 0, -5, 17, 21, 11, 11, 0, -11, 8}
}
, {{7, -10, -21, -6, 23, 12, 5, -8, -13, -12, -5, 12, -27, 9, -9, -4}
, {-1, 21, 11, 0, -19, -22, 15, 26, 15, -11, 12, 9, 0, 25, 25, -15}
, {-27, 12, -7, -7, -20, -8, 7, 18, -11, -10, -25, 14, -8, 11, -24, 11}
}
, {{18, 14, 19, 15, -1, -12, 6, 6, 15, -22, -21, -16, -3, -18, -3, -11}
, {19, 25, 16, -6, -11, -18, -19, -3, 8, -11, -19, -10, 25, 21, -17, 18}
, {11, -21, -15, 32, 11, 7, -10, -18, 18, -1, 6, -24, 9, -17, -3, 17}
}
, {{-8, -24, 3, -16, 1, 5, 2, -21, -1, -25, 10, -6, -7, -4, 11, 9}
, {0, -23, 20, 21, -22, 3, -22, -13, 21, -10, 6, -14, -9, 10, -10, -16}
, {22, 22, -2, 5, -24, 3, -3, 20, -4, 10, -13, 6, -22, -11, -31, 10}
}
, {{12, 14, -21, 24, 20, 18, -13, -4, -1, 1, -25, -25, 21, 30, 8, -19}
, {-14, 20, -27, 0, 1, 11, -22, 9, -11, 8, 25, -11, -4, 4, -16, -12}
, {0, 12, -7, -10, -12, 25, 8, -19, -30, -18, -24, -13, -25, -11, 7, 14}
}
, {{-12, 9, -15, 2, -8, 16, -12, -18, -2, -15, -15, 20, -22, -24, 5, -17}
, {22, 19, -25, 20, -14, -18, -17, 12, 4, -17, -2, 8, -18, -6, 16, -18}
, {1, 20, -23, -22, 8, -19, 2, 15, -24, -7, -9, 10, 16, 19, -22, 0}
}
, {{-13, 17, 17, -1, -6, 10, 14, 23, -19, -4, 19, 21, -2, 25, 12, 6}
, {-11, -10, -6, 14, 8, 10, -27, -15, 23, 25, -10, -24, 7, 1, 3, 25}
, {5, -14, -6, 0, -22, -9, 1, -6, -7, 16, -11, 8, 10, 1, -18, 24}
}
, {{-24, 11, -16, -10, 30, 22, -7, -10, 21, 0, -18, 4, -4, 27, 24, 4}
, {22, 19, -18, 20, 12, 19, 4, 19, 8, 3, -21, 21, 17, -1, -19, 6}
, {-3, 15, -2, -13, -24, 13, 25, 4, 21, -17, 0, -6, -15, 2, -7, -19}
}
, {{-8, -6, 3, 20, 2, 17, -23, 20, -13, 12, 28, -2, 13, 23, 1, -13}
, {17, 20, -12, -17, 19, 3, 15, 17, 14, 26, -3, 8, -3, 9, -14, -7}
, {-22, -15, -16, -24, 10, -9, 0, -23, -2, 20, 5, 18, -23, -10, 3, 24}
}
, {{-13, 3, -15, -7, 3, 7, 11, -20, -2, -4, -22, 14, 20, 4, -18, -25}
, {0, 3, -16, -16, -8, -15, 19, 14, 5, -12, -9, -21, 1, -24, 17, -28}
, {-11, -22, -8, -1, -28, -10, -23, 14, 9, -11, -25, -15, -2, -3, 23, -17}
}
, {{-16, 2, -21, 27, 12, -25, -14, -25, -22, 14, -17, -3, -10, 19, -19, 8}
, {20, 26, 11, 2, -4, 5, 18, -16, -16, -11, -26, -11, -8, 1, 20, 21}
, {10, -1, 26, 5, 3, 17, -21, 26, 6, 26, -7, -20, -8, 26, 18, -19}
}
, {{-1, -21, 4, -1, -12, -1, 8, -10, -20, -20, -19, -3, 22, -7, 23, 15}
, {-21, 4, -16, 0, 3, 19, 16, 10, -20, 4, 14, -25, 2, 4, 26, -12}
, {-21, 21, 12, -25, 16, 5, 2, -18, -7, -13, 19, 2, 1, -10, 22, 7}
}
, {{5, 14, 2, 25, -21, -24, -23, 5, 18, 9, 22, -4, -3, -20, 26, 17}
, {-19, 0, 8, -8, -23, 15, -25, -13, -3, -30, -9, -3, 3, 7, 4, 1}
, {-25, 16, -12, 19, -21, 20, 23, -13, -26, -21, -22, 8, 9, -9, -6, -5}
}
, {{-26, 9, 1, -25, 15, 3, 8, -23, -25, -24, -11, 14, -9, 12, -15, -4}
, {-16, -19, 11, -4, 4, 18, -12, 20, 4, 20, 9, -4, 13, -21, -19, -25}
, {21, -9, -7, 13, 18, 15, -5, 8, 4, 9, 8, -23, -11, -10, 18, -8}
}
, {{27, 8, 13, 20, -7, -14, -20, -6, -4, 23, -1, 19, -23, -10, -1, 11}
, {-1, 15, -17, -24, 18, 2, -5, 7, -8, 6, 26, 4, -2, -21, -23, 4}
, {-21, -20, 7, 18, -17, 11, 10, 22, 16, -2, -16, 5, -11, -11, -18, 0}
}
, {{-22, -13, 8, -17, 6, 14, 9, -10, -21, -27, -18, -3, -19, -30, -10, -15}
, {12, 0, 16, -8, -10, 1, -16, 14, -14, 20, 1, 5, 17, 23, -23, 3}
, {1, 13, 4, 2, 10, 22, 12, 29, 5, -8, -1, 19, -1, 14, 2, -6}
}
, {{19, -16, -14, -24, 21, 19, -8, 17, 0, -21, 4, -25, 6, -11, 8, -22}
, {-7, -6, -22, 22, -18, -1, 5, -28, 22, -11, -13, -3, -7, 6, 1, -2}
, {-12, 6, -10, -18, -2, -11, 6, 25, 24, 23, 11, 10, -18, -17, -8, 10}
}
, {{9, -23, -10, 0, -15, -16, 14, -11, -14, -19, 16, 9, 13, -12, -12, -18}
, {-17, -24, -8, 0, -11, 22, 15, -19, 21, -5, 16, -15, -1, -18, -11, -6}
, {-6, -23, 16, 24, 4, -16, -11, -5, 17, 5, 19, 7, -9, -14, -18, 25}
}
, {{18, -1, 1, -25, -12, -19, 3, -25, 0, 1, 21, 15, -10, -26, 13, -5}
, {8, 21, -23, -13, -15, 13, 7, 2, -20, -28, -1, -20, 13, -4, -16, 14}
, {22, -14, -24, -17, 20, 18, 18, -20, -19, -14, -8, 10, -2, -22, 14, -7}
}
, {{-23, -23, 9, -5, 17, 20, 17, -5, 20, -16, 12, 22, -9, 12, 13, -2}
, {-21, 21, -24, 15, 10, 8, 19, -16, -20, 10, -1, -21, 7, 0, 26, 24}
, {13, -5, 19, 10, 10, 8, -18, 13, -2, -17, 6, -13, 22, 15, 5, 1}
}
, {{-11, -22, 13, -17, -2, 8, 3, 1, 17, -14, 4, -20, 12, 4, 9, -20}
, {26, 9, 10, 2, 4, 20, -26, 15, 17, 1, 7, -16, 24, 5, -21, 23}
, {-21, 17, -13, -26, -12, 25, -5, -3, -26, 15, 16, 5, 23, -12, -11, 22}
}
, {{22, -26, 22, -20, 24, -18, 7, 9, -1, 6, -15, 5, -9, 6, 7, 23}
, {20, 3, 4, 18, 23, 29, -15, 13, -12, 5, -5, 16, 11, -22, 15, 19}
, {10, -17, -24, 8, 19, -19, -1, -11, -23, -10, 14, -6, 24, -6, 9, 20}
}
, {{3, -3, 10, -14, -4, 18, 15, -17, 21, 3, -1, 12, -1, 3, -18, -5}
, {-13, 17, 4, 12, 18, -21, -23, -6, -9, -13, 8, -22, -22, -2, 13, 23}
, {11, 18, -10, 18, 4, 23, 20, 7, 10, -12, 1, 0, 3, 20, 22, -3}
}
, {{-8, -20, -20, 13, 11, -24, -4, 0, -3, 14, -1, 0, 12, 5, -3, -6}
, {10, 18, -16, 1, -19, -1, 20, -22, 14, 0, 14, -6, -31, 15, -11, 19}
, {-8, -28, -19, 23, 9, 13, -8, 17, 7, 22, -15, 16, -14, 14, -17, -25}
}
, {{9, 5, -7, -3, 27, -20, -24, -25, -2, -10, -21, 7, 16, -22, 0, 17}
, {21, -3, -10, -23, -14, 5, 3, -23, 4, 18, 4, -23, 3, 11, -30, 6}
, {-4, 13, -5, 15, -20, -10, -23, -8, -5, -22, 26, 1, 21, 8, -6, -1}
}
, {{-20, 21, 24, -4, -11, -6, -13, -27, -1, -21, -8, 14, 7, 28, -4, 12}
, {-11, 0, 4, 5, 9, -11, 19, -20, -24, -22, -3, -31, 2, 23, 4, -4}
, {-2, -22, -22, 23, -9, 27, -11, 23, -10, -18, -7, 1, 0, 14, -14, 21}
}
, {{9, -8, -18, -12, -20, -20, 15, -22, -2, -20, -6, 12, 6, -2, 19, -3}
, {5, -6, 24, 13, -2, -5, -9, -24, -23, -27, 6, -28, 10, -5, -7, -26}
, {-1, 19, 23, -7, -5, 24, 9, 20, -12, -6, 21, -5, -23, -3, 12, 15}
}
, {{18, 25, -11, 16, -25, 28, -11, 13, 9, 15, 4, -8, 2, -9, 19, 25}
, {6, -4, -13, 1, 12, 15, 23, 15, -2, 17, -4, -12, 20, -18, 10, -11}
, {-15, 6, 25, 23, 18, 22, 13, 5, -12, 5, 2, -11, -28, 11, -7, 26}
}
, {{8, -11, 17, 17, -21, -18, 3, 5, 15, -22, 4, 2, 24, 2, 13, 16}
, {-11, 17, 15, -17, 5, 15, -29, -26, -24, -21, 8, -13, -23, -11, -1, -17}
, {1, -21, -28, 20, 3, 13, 12, 7, -22, -2, 20, -11, -1, 9, -16, 11}
}
, {{26, -22, 6, -24, 23, 12, -20, -29, -26, 16, 22, 27, 7, -13, 12, 18}
, {22, 1, -15, 24, 8, 21, -18, -21, -19, 16, -18, -24, -22, 19, 16, 29}
, {12, 17, -5, -14, -15, -4, 26, -27, -25, 17, -24, 20, 11, 21, 4, -19}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_268_H_
#define _BATCH_NORMALIZATION_268_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       111

typedef int16_t batch_normalization_268_output_type[111][32];

#if 0
void batch_normalization_268(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_268_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_268_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_268.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       111
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_268(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_268_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_268_bias[32] = {69, -47, -17, -2, -13, 82, 21, 74, -84, -119, -85, 122, -33, -1, 60, 22, -15, -17, 41, 69, 65, -95, -65, -70, -107, 7, 62, 36, 35, -143, 37, -38}
;
const int16_t batch_normalization_268_kernel[32] = {292, 259, 254, 289, 258, 279, 215, 209, 259, 250, 250, 190, 257, 222, 253, 281, 332, 258, 291, 275, 218, 208, 284, 168, 266, 223, 233, 266, 229, 197, 279, 197}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_344_H_
#define _CONV1D_344_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       111
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_344_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_344(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_344_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_344.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       111
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_344(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    32
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_344_bias[CONV_FILTERS] = {-1, -1, -1, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0}
;

const int16_t  conv1d_344_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-8, -11, 0, 21, -27, 21, -16, -22, -3, 22, -10, -20, 5, -19, -9, -18, -21, -10, 1, 17, 7, -27, -19, -15, 2, 4, 19, 1, -14, 13, -10, 24}
, {-21, -17, -13, -3, -29, 21, 18, -5, 20, 19, -11, -20, 4, 19, -4, 3, -15, -9, 2, 11, 6, 17, -25, 18, -4, -7, 12, 0, 24, 3, 12, -6}
, {-11, -9, 15, 29, 14, 27, -28, -7, 1, 23, -25, 1, 24, 14, -18, 5, 13, 8, -2, -12, -15, 2, 2, 4, 21, -11, 22, -10, 4, -6, 17, 1}
}
, {{-13, 22, 3, 18, -12, 18, 21, 1, -25, -1, -12, 7, -18, 25, -6, 10, -13, -2, 9, -26, -12, 3, -6, -13, 0, -9, -14, 5, -6, 6, -13, -8}
, {4, -11, -25, -6, -20, -3, 1, 11, 25, 11, 13, -21, -6, 9, -16, -12, 26, -28, -16, 4, -7, 10, 19, -21, 27, 23, 7, 27, 20, -13, 17, -5}
, {-9, -8, -8, -7, 12, 23, -20, -15, 3, -12, -16, 13, 9, -13, 19, -9, 10, -8, 1, -18, 9, -14, -20, -7, -6, -2, 13, -20, -21, 11, 3, -17}
}
, {{-18, 4, -2, 6, 23, -14, 19, -11, -10, -6, -2, -28, 24, -20, 9, 7, 22, 16, 20, -12, 23, 3, 4, -5, 11, -15, -3, -13, 18, -18, -15, -16}
, {19, 22, -20, -3, -17, -17, -21, -6, -16, -12, 6, -14, 3, 26, -8, 4, -31, -22, -19, 11, 1, 1, 12, -25, 26, -11, -21, -20, -24, -22, -13, -21}
, {-10, -1, 18, 2, 7, 23, -18, -17, 25, -9, 18, 1, -24, 15, 4, 9, 4, 25, 20, -6, 1, 1, 4, 10, 15, 29, -26, -14, 18, 22, -20, -7}
}
, {{24, -24, 3, -23, -28, 23, 24, -15, 5, -21, -6, -8, -3, 14, 6, -2, -13, 10, 15, 0, -1, 8, 15, 19, 14, -11, -19, 8, 16, 22, 23, 13}
, {-4, 12, -6, 12, 23, 14, -21, 22, 10, 2, -10, 28, -15, -12, 20, -10, -24, -7, -9, -12, 10, -1, -9, 0, -5, -19, -24, -13, 9, -27, -14, -19}
, {26, 14, -5, -16, -14, -1, 9, -1, -13, 1, -10, 25, 13, 21, -13, 18, 2, -9, 23, 24, 5, 24, -18, 6, 4, -22, 0, -12, 0, -21, -15, -1}
}
, {{3, -1, 5, 9, 8, -23, 6, -24, -8, 6, 20, 13, 3, 9, 7, 4, -11, -21, 25, 25, -22, -17, -8, 11, 14, 13, -24, 1, 20, -23, 3, -11}
, {15, 4, -3, -17, -1, 22, 9, 15, 5, -9, 0, -17, -8, -23, 23, -2, 14, -11, -26, -24, -28, 18, 1, -8, 7, -24, -25, -24, 1, 21, -3, -10}
, {20, 10, -16, -4, -23, 17, 23, 10, 3, 14, -3, 9, 22, 16, 19, 24, 8, -11, -13, -2, 21, -22, -7, 19, -10, 17, -15, -11, 20, -27, 3, -15}
}
, {{8, -10, -19, 17, 10, 2, 15, 2, 10, 1, -21, -17, -27, -21, 18, -6, -4, 6, 25, 20, -11, 10, 15, 9, 26, 21, 4, 8, 17, 23, -12, 18}
, {12, -23, 28, 12, 14, 5, 25, -6, 10, -18, 6, -5, -6, -25, 24, 6, -10, 18, -23, 20, 2, -18, -7, -25, -25, -28, -23, -7, -18, 20, 4, 1}
, {-17, 9, 7, 19, 24, -19, -4, 23, -27, 15, -17, -19, 13, -11, -8, -7, -2, 6, -30, 28, 21, -5, 13, -10, 25, 15, 1, -15, 14, -2, -9, -7}
}
, {{-14, 25, -3, 2, -22, -2, 11, -15, 11, -18, -19, 7, -17, -7, -15, 16, 10, 19, 19, 10, -13, 25, 12, 22, -2, 11, 2, 23, -6, -16, 22, -13}
, {-5, 4, 14, 9, 5, 25, 19, 15, 11, 13, 22, 9, -24, -3, 14, -23, -2, -17, 20, -3, -7, -6, 21, -1, 3, 18, 17, -7, 24, -15, -16, 14}
, {-19, -10, -20, 19, 12, 23, -5, 15, -9, -2, 4, 5, 26, 7, -25, -13, 4, 8, -27, -7, 19, 4, 21, 17, 8, 11, 24, 4, 12, 0, 28, 3}
}
, {{-22, 19, -26, 27, -7, 6, -6, -11, 23, -19, 18, -24, -19, 16, 10, -10, 30, -17, -11, 11, 1, -18, -5, 18, 6, -3, 10, 11, 19, 0, 16, 17}
, {-23, 21, 21, 11, 3, -13, 5, -5, 14, -3, -20, 17, -19, 4, -26, -5, 22, 28, -16, 10, 2, -13, 21, -11, 24, 0, -7, -5, 15, 13, 17, 0}
, {-15, -22, -21, 12, 10, 10, -13, -11, -11, 26, -5, -22, 6, -4, 11, -20, 20, 21, 21, 4, -10, 3, -1, -16, -23, 27, -3, -21, 24, 26, 16, -14}
}
, {{23, -4, -8, -18, 7, -7, 0, 11, -24, 20, 2, 16, 19, 13, 20, 18, -8, 21, 21, -20, -12, 21, 10, 5, -16, 21, 15, 23, 2, -24, -22, 1}
, {2, 11, -21, -9, 20, 17, 18, 11, 25, 19, 12, 23, 21, -22, 24, 12, 21, -10, -20, 7, 4, -16, -25, -16, -24, 23, -2, 23, -6, -23, 15, -6}
, {1, -12, -1, -6, 13, -2, -27, -2, -13, -8, 15, -13, -20, 18, 10, -3, -11, 1, 1, -6, 22, -22, -17, -4, 7, -8, 8, 11, 8, 22, -12, 2}
}
, {{20, -10, 14, -23, 12, -9, -22, 20, 21, -11, 26, 9, 15, 7, -2, 21, 23, 11, -28, 5, 14, 12, 16, -1, 9, -24, 24, -31, 17, -15, -12, 26}
, {-16, -22, 12, 1, -3, 2, 13, -19, 18, 11, 5, 0, -23, 21, -8, -6, 6, -7, -5, 11, 5, 18, -11, -1, 0, 9, -23, -22, -1, -1, -1, -21}
, {-5, 8, 5, -19, -7, -20, 16, 3, 12, -5, 21, -18, 1, 7, -2, -4, -5, -20, -25, 11, -24, 12, -6, 3, -2, 5, 16, -7, -25, 15, 5, 12}
}
, {{-9, 11, 17, -11, -13, 18, -15, 13, 5, 25, 2, -12, -24, 13, 1, 22, -10, 29, 6, 8, -5, 16, -25, -25, 18, -3, -21, -6, 12, -23, -20, -4}
, {20, -20, -22, -10, -12, 22, -22, 3, 24, 20, 6, -8, 6, 12, -27, 18, -22, -8, -14, -17, -4, 0, 29, -17, 5, -12, 19, -13, 12, 22, 21, 1}
, {-3, 12, 6, 24, -13, 16, -5, -7, -24, -15, 25, 22, -13, -5, -19, -17, 9, -20, -6, -10, -8, -24, 22, 15, -3, -11, -20, -22, -13, 22, 24, -22}
}
, {{-13, -24, 23, -19, -7, 14, 4, -15, -4, -22, 10, -17, -9, -10, -6, -8, 20, 7, -23, 13, -13, -9, -26, 2, -22, 15, -27, -13, 2, -20, -17, -20}
, {-2, -15, -6, -16, 11, 3, 10, 20, 23, 24, -15, 26, -25, -13, -12, 22, -20, -13, -14, -4, 1, -15, 18, -9, 7, -6, -26, 5, -15, -3, 18, 12}
, {-21, 10, -10, 0, -13, 24, -4, -8, 5, 18, -12, 14, -4, -11, -21, -17, -24, -21, 18, -4, -19, -12, -23, 12, -5, -21, -15, 22, 25, 16, -8, 21}
}
, {{19, -23, -4, -11, 16, -3, -19, -12, 23, 22, -7, 19, 12, -10, -3, 21, -21, -21, -9, 24, 19, -2, -14, -17, -24, 27, 20, -5, -7, -6, -10, 4}
, {-18, -3, -13, -18, 17, 7, -6, 16, 15, -16, -1, -17, -12, 17, -4, 20, 8, 6, -20, -7, -3, 14, -5, -2, -9, 7, 17, -23, -18, -10, 12, 4}
, {-32, -20, 14, -8, -24, -15, -21, -20, -18, 1, -26, 1, -9, -24, 23, -28, -3, -8, 0, -2, 6, -27, -26, 19, -18, 12, 0, -12, -28, -15, 12, -5}
}
, {{20, 23, -6, -1, -15, 27, 8, 18, 6, -23, 4, 4, -5, 23, -17, 11, 26, -18, -16, 20, -13, 20, 17, -19, -3, 15, -28, 15, 2, -8, -25, 25}
, {-14, 8, 16, 11, 10, -12, -4, 2, -25, -19, -8, -2, 22, 25, -19, 2, 0, 13, -24, 8, -3, 14, -26, 11, 2, -12, 15, -16, 15, 0, 12, 28}
, {-18, -9, 16, 10, 14, 5, 3, 18, -6, 6, -25, 6, -3, 7, -2, -24, 29, 11, -22, -7, 14, -26, 16, -12, -22, 7, 24, 12, 26, 10, 22, -23}
}
, {{-19, -25, 6, 29, -17, -20, 14, -17, -15, -21, -24, -3, -18, 2, 13, 11, 14, -5, 5, -14, 22, 17, -16, -20, -24, 17, 2, 2, -14, -15, -15, 20}
, {14, 11, 26, -22, -25, -23, 21, -6, 4, -16, 11, 5, 27, -16, 5, 25, -17, 1, -23, 15, -5, -23, -22, -11, -7, -13, 0, 15, 2, -10, -22, -2}
, {-4, -1, -23, -11, -15, 7, 5, -18, 8, -16, 12, 17, -14, 18, 12, 5, -3, -22, 25, -15, 25, -12, -3, 5, 4, 19, -10, 24, 18, -24, 23, -16}
}
, {{10, 9, 25, -17, -16, -3, -3, -15, -6, 2, 11, -10, -14, -4, -1, 13, -2, 19, -11, 7, 22, -27, -25, 8, 16, -24, -15, -4, -5, 2, 6, -7}
, {1, -14, -14, -23, 11, 15, -25, 1, 8, -5, -8, -7, 19, 3, 24, 26, 8, -9, -16, -13, -16, -11, -6, -14, -9, 12, -11, 5, 17, -10, 10, 27}
, {-12, -9, 6, -3, -8, 19, -6, 5, -6, 8, -1, -19, -4, -12, -5, 7, -15, -9, -13, 0, 17, -17, 23, -18, -2, -22, 19, 2, 24, 12, 14, -21}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_269_H_
#define _BATCH_NORMALIZATION_269_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       109

typedef int16_t batch_normalization_269_output_type[109][16];

#if 0
void batch_normalization_269(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_269_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_269_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_269.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       109
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_269(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_269_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_269_bias[16] = {-10, 47, 21, -28, -34, -67, -131, -57, -116, -26, 17, 108, 72, -83, 38, 35}
;
const int16_t batch_normalization_269_kernel[16] = {175, 173, 175, 179, 175, 185, 139, 111, 198, 145, 170, 150, 107, 156, 113, 227}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_345_H_
#define _CONV1D_345_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       109
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_345_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_345(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_345_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_345.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       109
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_345(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_345_bias[CONV_FILTERS] = {0, -1, -1, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0}
;

const int16_t  conv1d_345_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{22, -22, 12, -19, 13, 23, 11, 6, -20, 16, -1, 12, -3, -9, -25, -15}
, {26, 17, -13, -10, 6, 0, 22, -11, -23, -6, 12, 14, 20, 14, -1, 5}
, {-7, -30, -7, -17, 12, 27, 26, 8, -4, -7, -29, -19, -14, 6, 0, 13}
}
, {{0, -18, -8, 12, 20, -5, 19, 7, -27, -8, -6, -22, 14, -18, 21, -16}
, {-17, -9, 12, 25, -25, 10, 19, -18, 1, -8, -19, -18, -13, 3, 7, -5}
, {-1, 3, -24, 24, -25, -13, -17, 24, 9, 7, 27, -25, -8, -23, -21, -5}
}
, {{13, -21, 17, -18, -9, 15, -22, -24, -3, 15, -1, 22, 2, -17, 20, -7}
, {0, 19, 15, -15, 25, 7, -18, 13, -19, 21, -2, 4, 25, -25, -18, -2}
, {-9, 3, -18, -6, 18, 17, 22, -21, -23, 15, -25, 8, 13, 6, 18, -6}
}
, {{9, -22, -14, 2, 7, 0, -11, -1, 0, -15, 21, -13, -10, 16, 1, 8}
, {-10, 23, 5, 13, 19, 21, 12, 10, 2, 9, -13, 5, 21, -4, 10, 13}
, {12, 20, 0, 12, 9, -24, 1, -2, 23, 22, 16, -5, 26, 8, -12, 18}
}
, {{22, -11, -13, -17, -25, 12, 5, 7, -5, -15, 0, 14, 13, 4, 23, 18}
, {-22, -4, 19, 20, -3, 18, -1, -6, -8, 10, -8, 0, 0, -4, -6, -10}
, {-28, -14, -4, 22, -25, -19, 24, -14, -21, 11, -16, -7, 4, 9, 2, 10}
}
, {{14, 17, 7, -10, 20, 5, -10, -7, 5, -11, 25, 9, 8, -2, 23, 18}
, {-13, -16, 9, -1, -21, -12, -7, 23, 25, 17, 20, 7, -8, -17, -15, 16}
, {18, -13, 25, -22, 12, -26, -5, 20, 8, 25, -7, 23, -4, 17, 23, -15}
}
, {{-12, -7, -20, 10, -8, 2, 21, -22, 18, -21, 11, -17, -11, 22, -31, 6}
, {14, -13, 6, -22, -23, -26, 6, 24, -8, -3, 25, 9, -25, 7, 14, 19}
, {-8, -12, 8, -22, -26, 1, -11, 5, -21, 20, 17, 18, 0, 11, -6, -21}
}
, {{-3, 4, 15, -3, -22, -20, 16, 6, -7, -14, -20, -21, 2, 1, -22, -14}
, {-19, -15, -8, -18, -14, -9, -7, 25, 23, -14, 5, 10, 19, 6, -22, -23}
, {29, -8, 4, 7, 6, 0, 9, 4, 18, 7, 7, 17, -20, -8, 18, 2}
}
, {{-15, 16, -5, 16, -16, -21, 4, -6, 0, 4, -5, -3, 16, -19, 7, 25}
, {8, -7, -22, -10, 0, -1, 8, -16, -11, 3, 25, 28, -17, 5, -16, -16}
, {-18, 4, 1, 17, 7, -4, -1, 16, -16, 20, -7, -18, -14, 17, -24, -15}
}
, {{-15, 12, -17, 13, -20, -11, -8, -7, -3, 10, -18, -1, 12, -11, -8, -23}
, {-21, -20, -9, -12, 5, 14, -29, 0, -15, -11, -11, -5, -15, -10, -19, -7}
, {-14, -1, 16, -10, 7, -10, -27, -22, -21, -24, -11, -11, -10, -22, -16, 17}
}
, {{-1, -27, 13, -17, -25, -15, -23, 16, 23, -1, -15, -18, 32, 6, -22, -4}
, {11, -16, -14, -2, -5, 11, 0, 4, -27, -20, -28, 12, -10, -22, -8, -14}
, {-10, -5, 18, -21, -21, -2, -16, 7, -9, 2, -22, 13, -10, 18, -14, -25}
}
, {{-13, 0, 11, 12, -10, 25, -17, -27, 19, -19, -12, 3, 1, -5, 9, -23}
, {-12, 1, -8, 15, 5, -6, -27, -12, -1, -23, -12, -26, 1, -8, -2, -5}
, {-8, -6, -4, -26, -10, 20, -10, 25, -11, -14, -27, -14, 5, 27, -15, 18}
}
, {{-18, -8, 11, 19, 3, 3, 15, 7, -1, 13, -3, -19, -3, -12, 19, 15}
, {12, 7, 12, 22, 11, 0, -1, -12, -15, -25, 6, -7, -8, 15, -18, -15}
, {10, -14, 4, -4, -19, 23, 18, 14, -13, 18, 0, -23, 27, 28, -14, 4}
}
, {{21, -10, 2, 26, 7, -2, -28, -17, 0, -23, 13, 4, -7, 7, 17, -19}
, {13, 14, -3, -18, -7, -19, 1, -12, 23, -5, -1, 28, -19, -6, 0, -8}
, {-3, -17, -18, -11, 10, 3, 6, -1, 22, 4, 19, 15, 8, 1, -24, -25}
}
, {{16, 0, 23, 15, 17, 22, -3, -10, 19, -22, -10, -18, -23, 19, 14, 3}
, {-21, -10, -12, 3, 0, -13, 13, 22, 11, -13, 21, 14, -32, 21, 28, -18}
, {24, -17, -10, 16, 6, 6, 14, -19, -15, 12, 6, -10, 12, -12, -2, -10}
}
, {{-11, -11, -12, -3, 8, -4, -23, 24, -22, -13, 4, -11, -19, -25, -12, 14}
, {-1, 23, -19, -13, 6, -3, 24, -4, 2, 24, -11, -6, 3, -8, -3, -21}
, {-5, 15, -3, 2, -9, -13, -9, 9, -24, -26, 3, 24, -13, -20, -6, -18}
}
, {{19, -4, -11, 11, 12, -14, 29, 19, -2, 3, 6, 14, -19, -2, -22, 15}
, {-4, 4, 24, -17, 2, 17, -21, 13, 15, -2, 5, 16, -10, -1, -28, 22}
, {-22, 11, 6, -22, 17, 24, 19, 23, -20, 22, 21, -7, 6, -20, -21, -7}
}
, {{21, 17, 22, 16, -16, 3, -10, 0, 3, -10, 7, -19, -12, 15, -18, -13}
, {22, 7, -8, 18, -3, 26, 15, 11, 8, 12, 20, 7, 13, -15, -9, -20}
, {25, -9, 9, 9, -21, 20, 20, -15, 4, 18, 7, 18, 4, -8, 15, -11}
}
, {{11, 2, 16, 12, 10, -4, 13, -15, 4, 15, -9, 18, 20, -11, -9, 25}
, {18, -2, -18, 5, -2, 14, 14, 8, -15, -22, -18, 7, 21, -13, 13, 8}
, {1, -8, 9, 29, 19, -15, -16, 4, 10, 11, -26, 12, -30, 7, -3, -11}
}
, {{-5, -3, 16, -8, 18, 22, 14, 12, -8, -19, 21, -9, 23, 9, -12, 24}
, {23, 1, -4, -22, -4, -23, -5, 4, 5, 16, 21, 15, 16, -13, 22, -13}
, {-2, -9, 26, 1, -7, 13, 16, 25, -12, -22, 24, 0, 17, 15, -27, 21}
}
, {{12, -17, -5, -6, 19, -19, -25, 13, 0, -28, 7, -16, -9, -24, -5, -4}
, {10, -27, -17, -9, -20, -27, 16, -1, -16, -20, 5, -19, -6, -19, 3, -1}
, {22, -10, 20, 9, -8, -12, 22, 14, -15, 15, -18, 20, -9, 8, -1, 19}
}
, {{5, 3, 21, 9, 8, 11, 2, 11, 30, 5, -10, -10, -16, -15, 18, -4}
, {-25, 16, 8, 17, -10, 1, 4, -9, 26, -11, 23, -8, 21, 13, -24, -21}
, {-19, -23, 20, 8, -20, -2, 24, -20, 29, 22, 24, 24, 7, 19, -10, 12}
}
, {{-4, 25, -11, -16, -13, 2, -7, 6, 5, 1, 9, 27, -5, -28, -8, -21}
, {15, 10, -28, -2, 21, -21, 5, 1, 18, 8, 17, -10, 2, 20, -2, -14}
, {-6, -10, -12, 11, 6, 19, -8, 17, -24, -12, 9, -7, -3, 4, -23, 7}
}
, {{-19, -28, -17, 26, 15, 19, -9, -7, 13, -4, 17, 7, 2, -14, 20, 2}
, {-19, -24, 0, -23, 17, 20, -13, -11, 13, 12, -9, -25, 13, 6, 3, 0}
, {-25, -10, 1, -15, 25, -18, -16, -8, 15, -11, -5, -12, 2, -12, 8, 15}
}
, {{20, 8, -6, 3, 28, -7, 13, -19, -21, -3, -11, -21, -25, -1, 16, -10}
, {-16, 12, -11, -17, -22, -19, -4, -11, 3, 16, -4, -23, -25, -4, 22, 17}
, {14, 10, 21, -3, -4, 15, -17, 7, 3, -12, 3, -20, 12, -11, -16, -14}
}
, {{13, -7, 17, 18, -19, -22, -6, 10, -26, 24, -18, -21, 1, -1, 6, -13}
, {9, 18, 7, 1, -6, 25, -14, -23, 10, 2, -17, -18, 12, 18, -7, -1}
, {-22, 11, -25, 14, 12, 3, 19, 9, 16, -27, 19, 12, 19, 10, 10, 22}
}
, {{4, 21, -16, 14, -7, -2, 0, -14, -30, -17, 24, -1, -27, 16, -7, -23}
, {20, -5, -4, -6, 28, 8, 23, -20, 3, 20, -11, 0, 17, 25, 20, 25}
, {7, 0, 8, -19, -15, -19, -4, -18, -20, -10, 21, 22, -21, 9, -1, -6}
}
, {{16, -20, 11, 15, 25, 21, -15, 19, 14, 26, -3, 2, 1, -9, 13, -21}
, {-19, 20, -13, -23, 2, -13, 21, -8, -8, 1, 2, 13, 0, -6, -17, 4}
, {11, -7, 29, 2, -13, 24, -17, -1, -15, -12, 12, 6, -4, -22, -13, 13}
}
, {{-26, 21, 6, -17, -17, 17, -21, 22, 27, 10, 17, -18, 24, -3, 4, -21}
, {18, 22, -9, 5, 9, 0, -3, 24, -12, -9, -13, 20, 14, -1, -18, -20}
, {2, 3, 1, -10, -3, 21, 15, 1, 21, -2, 20, 20, -15, 9, -18, -22}
}
, {{-27, -22, 4, -25, 7, 25, 26, 23, -27, 2, 25, 21, 2, -17, -24, -20}
, {7, -10, -4, 0, -12, 5, 1, -5, 2, 18, -20, -16, 8, -19, 0, 4}
, {28, -9, -27, -1, 5, 23, -14, 2, -1, -24, 19, -24, 15, -18, 17, 15}
}
, {{-10, 6, 12, 12, 20, -19, 20, 2, 10, -7, -25, 10, 19, -28, 21, 1}
, {8, 6, 13, -13, 24, 5, -15, 8, 0, 3, 23, -11, 24, -8, 11, 14}
, {12, -2, 9, -25, -20, 24, -4, -20, -16, 9, 23, -11, 6, -23, 10, -2}
}
, {{2, -8, 12, 24, -13, -18, -6, -20, 14, -4, -19, -4, -4, -20, -19, 1}
, {-28, -8, 19, -25, 4, -20, -21, 2, 17, -1, -10, -11, -10, 17, -7, 2}
, {17, -17, -17, -12, 18, 12, 20, -13, -17, -8, -7, 8, 12, -11, 10, -8}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_270_H_
#define _BATCH_NORMALIZATION_270_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       107

typedef int16_t batch_normalization_270_output_type[107][32];

#if 0
void batch_normalization_270(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_270_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_270_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_270.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       107
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_270(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_270_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_270_bias[32] = {-18, 82, -16, -111, -1, -100, 43, 34, 47, 94, 117, 64, -42, -2, -50, 102, -43, -75, -94, -79, 80, -88, -3, 28, 64, -51, -22, -31, -55, 24, -77, 64}
;
const int16_t batch_normalization_270_kernel[32] = {164, 276, 197, 177, 249, 179, 218, 303, 294, 108, 166, 144, 147, 251, 178, 210, 140, 142, 246, 143, 268, 172, 295, 183, 263, 226, 245, 251, 234, 217, 232, 196}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_346_H_
#define _CONV1D_346_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       107
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_346_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_346(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_346_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_346.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       107
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         2
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_346(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    32
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_346_bias[CONV_FILTERS] = {0, -1, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, -1, 0, -1, -1}
;

const int16_t  conv1d_346_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{11, -4, 23, 3, 4, 20, 20, -18, 15, 26, -17, -18, 11, 9, 2, -17, -13, 9, -2, 19, -5, 0, 20, 6, 22, 21, -8, -24, -10, -25, -8, 24}
, {-26, 2, -5, -21, 24, -15, -21, -19, -25, 13, 11, 0, -12, 1, -19, -9, -24, -14, -20, 1, 20, -17, 16, 2, 14, 12, -4, -22, 0, 26, 3, -20}
, {-16, 19, -22, 19, -9, 6, 10, -3, 1, -13, 17, 7, 12, 21, 8, 5, -18, 13, 14, -10, 22, -19, 12, 18, 6, -21, 0, 4, -10, -7, 21, 2}
}
, {{9, -4, -2, -18, 10, 23, 1, -32, -6, -25, -3, -20, 0, 13, -7, 22, -3, 12, -10, -6, 12, 22, -20, 14, 21, 6, 22, 12, -1, 12, 15, -5}
, {-4, -4, -3, 14, 21, 1, -16, -14, 19, 1, -18, 11, -7, -16, -5, 7, -19, 17, -6, -2, 7, -10, 7, -20, 6, 16, 24, -8, -14, 9, 2, 11}
, {12, 6, 25, -25, 10, -6, -16, -6, 0, -12, 18, 13, 14, -6, -16, 3, 1, 13, -27, 9, -6, -23, 0, 1, -5, -11, -21, -3, 14, 2, 8, -12}
}
, {{-18, 26, -1, -5, 20, -13, -24, 23, -22, -19, -9, -25, -19, -18, 7, -1, 13, -16, -3, -11, -1, 13, 17, 12, 10, -22, 16, 24, 16, -13, 3, -7}
, {20, 23, -16, -2, 6, 15, 12, 13, -11, -5, -32, 12, -9, 14, 23, -22, 5, -21, -3, 24, -11, -21, 4, 14, 14, -2, -21, -12, 14, 10, -12, -11}
, {-25, 3, -17, -4, 15, -13, 4, 3, 19, 15, -16, 10, 16, -16, -20, -27, 25, -7, -19, -16, -19, 7, 1, -1, -24, -7, -13, -19, -24, 21, -3, -20}
}
, {{12, 16, -2, -10, 4, 22, -10, 1, 8, -20, -24, 4, -20, -5, -11, -16, 23, 9, 6, -20, 8, 24, -24, -18, 7, -25, 26, -23, -2, 8, 7, -20}
, {8, 12, -5, 4, 11, 25, -16, 4, 3, 16, -16, 18, 18, -16, -24, -4, 23, -2, -6, 16, 6, 7, 2, -18, -16, 24, -5, 23, -16, -18, -17, 14}
, {-24, 14, -8, 24, -2, -2, -11, 11, 0, -12, 3, -26, 4, 14, -5, -14, 2, 2, 7, -6, -22, 23, -8, 23, -9, -12, -21, 12, 20, 23, 25, -12}
}
, {{4, 21, 10, -25, 0, -17, 0, 12, -11, 5, 17, -16, -4, -3, 6, -25, 18, -20, 10, -12, 11, 12, -17, 8, -23, 21, -21, -9, -11, -25, -9, -10}
, {22, 16, 16, 7, -7, 4, 18, 12, -11, -24, 0, 10, -4, -16, -10, -6, -25, -9, -7, 0, -11, -5, -5, -23, -24, -23, 2, -5, -15, -12, -24, -12}
, {-5, -14, -10, 6, 24, 24, -5, 14, 1, -10, 3, -2, 7, -13, -26, -1, -14, -17, -10, 5, -7, -8, -26, -4, 21, 10, -3, 14, -7, -10, -14, -15}
}
, {{-10, 18, 0, -12, -24, -3, -22, -25, -2, -5, 17, -2, 1, -22, 15, -4, -16, -3, 13, -19, 25, -12, -6, 14, 19, -6, 10, 3, -12, -2, -7, -4}
, {16, 23, 10, 23, 19, 6, -17, -18, -17, -7, 7, 8, -18, -12, 21, 6, -9, -3, -3, 5, -2, -19, -1, 1, -24, 22, -2, -2, -16, -17, -26, -15}
, {24, 5, 1, 3, 16, 25, -19, -1, -5, 11, -3, 8, -4, 0, 3, -10, -27, -1, 19, -2, -24, -7, -25, 22, 3, -9, -13, 15, 14, -16, 9, 15}
}
, {{-18, -22, 22, 6, 1, -22, -26, -23, -4, 12, -10, 18, 3, -3, -3, 1, 24, 15, 10, 20, 11, 0, 25, 5, 3, 6, 13, 3, -7, 9, 21, 9}
, {-20, -19, 18, -5, 16, -21, 20, 7, -6, -12, -9, -12, 0, 14, -25, 23, 25, 13, 21, 5, 2, 21, -2, 19, -3, -24, -15, 10, -4, -20, 23, 2}
, {-10, -17, 25, -23, -11, 6, 9, -18, -10, 5, 0, 6, -14, 6, -2, -14, -21, -23, -2, 3, -9, 1, -22, 11, 1, -20, 13, 14, -3, 17, 0, -4}
}
, {{13, -16, -5, 23, 19, 0, 24, -3, -27, 2, 22, 6, -23, -15, 12, 10, 15, 4, -6, 14, 0, -6, 26, 0, -16, -16, -12, 10, -20, -7, 8, -12}
, {8, -6, -29, 8, 13, -23, -5, -21, -14, -1, 26, -18, 5, 11, 25, -5, -13, -14, -19, -17, -14, -20, -13, 19, 21, -16, -3, -7, 7, -17, 12, 8}
, {22, 19, 20, 10, -7, 20, 1, -4, 20, -13, -12, 9, -2, 11, -26, 27, -7, -7, 9, 24, 23, -13, -9, 22, 2, 21, 12, 7, -13, 17, -9, 1}
}
, {{11, -6, -20, 14, 19, 14, 8, -5, -14, -23, 20, -21, -5, -11, 16, 2, -1, 2, -18, 11, 4, -4, -17, 5, 23, -25, -8, 18, -16, -2, -10, -7}
, {-2, -3, 12, -15, 8, 1, 18, 16, -4, 9, 18, -21, -13, -19, -17, -6, 22, 25, 0, -18, 8, 4, 15, 2, -9, -25, -17, -18, 4, 8, -15, -14}
, {-25, 19, 0, 25, 10, 2, 14, -19, 6, 16, -1, 10, 16, -3, -10, 3, -7, 24, -9, 7, 10, 3, 25, 9, -25, 10, 16, 8, 20, 12, -17, -16}
}
, {{9, 2, 7, 11, -15, 18, -25, -9, 15, -18, -10, 21, -24, -22, 19, -1, -5, -21, -18, 21, 21, -22, 12, 18, -25, 21, 13, -11, -20, -12, -13, -20}
, {17, 19, 7, 23, 12, -2, -3, -19, -4, 4, 19, 21, 19, -28, -26, -23, 8, 15, -18, 9, 16, -20, 9, -22, -5, 17, 12, 23, -4, -13, -14, 2}
, {-8, 28, 0, -1, -16, -1, 0, -15, 9, -25, -15, -17, -8, 18, 7, -8, -25, -18, 21, 6, 22, -23, 14, -9, 13, 22, -19, -20, -5, -10, 9, -17}
}
, {{11, -5, -3, -18, 21, -17, 23, 17, -5, -5, -19, 1, -25, 3, 7, -25, 23, -6, -4, -22, 17, -5, -25, -26, 30, -10, -3, 25, -12, 8, 3, 4}
, {-15, 17, 11, -3, 7, -4, 8, 25, -19, -10, -26, 10, -17, 24, 22, 5, 17, 21, 12, 10, 19, -25, 0, -27, -6, 6, -16, -14, -27, 16, 25, 22}
, {14, -12, 19, 19, -20, -6, 0, 6, 7, -11, 2, 9, -13, 0, -22, -23, -19, -9, -22, 25, -12, -1, 13, -14, -24, -14, -1, -26, 28, 21, -20, 4}
}
, {{-16, -23, -12, 21, 0, -4, 21, 17, 13, -8, 0, -2, -18, -28, -12, -17, -12, -14, -6, 2, -22, -3, 19, -14, -18, 8, 12, -11, -17, 31, -11, -4}
, {-11, 0, 20, -8, 22, -25, 14, -5, -1, 16, 5, 9, -21, 10, 2, -2, 13, -24, -11, -6, -7, -11, 2, -16, 14, -14, -10, 19, -19, 15, -14, 7}
, {-23, 15, -28, 6, -21, -18, 5, -8, -2, -5, -1, 19, 4, 27, 26, 5, 17, -22, 11, 21, 0, -8, 24, -12, 16, 10, -5, -21, 14, 23, -6, -13}
}
, {{16, 1, -4, 9, -23, -2, 26, -11, -22, -5, -1, 3, -5, 19, 9, 1, -26, -3, 18, -15, -15, 21, -22, -5, -11, 24, -23, -28, -10, -30, -3, 18}
, {-25, -21, -2, -26, -11, -7, -21, 14, 13, 4, 10, -24, 4, 20, 17, -24, -20, -10, 22, 19, 16, -17, -17, 2, 15, -22, 6, 9, -7, -25, -9, 21}
, {0, -20, 12, 0, 4, -28, -21, -4, 2, 2, 4, -17, -13, 13, -18, -24, 1, -4, -15, 17, 13, 7, 2, -12, 12, -25, 6, 15, 5, -21, 16, -10}
}
, {{-6, 10, -20, -12, -20, 16, -13, -11, -12, -23, -5, -5, -10, -1, 22, -16, 13, -5, -23, -18, 12, 1, 9, -16, -21, -23, -25, 11, 23, 25, -9, -15}
, {-10, -7, 4, 1, -7, -14, -28, 0, -3, 8, 12, -18, 18, 4, 3, 18, -10, 8, 10, 4, -11, -12, 21, -21, 1, -27, 10, 4, 3, -25, 14, -1}
, {-9, -16, -1, 22, -14, 8, 18, -5, -1, -20, 19, 10, 14, -7, -10, -2, 19, 13, -2, -15, 17, 0, 22, 4, -19, -14, 11, -12, -20, 20, 9, 18}
}
, {{-8, 2, -4, 25, 13, 15, -15, 11, -19, 8, 26, -17, 18, 4, -1, -8, -1, -13, -9, -21, -16, 6, -6, -27, -8, -18, 19, 17, 24, -13, 24, -5}
, {2, 18, -17, 21, -22, -10, 10, -19, 25, 5, -18, 23, 8, 25, -18, -7, 13, 12, -3, 21, -13, 11, 1, 15, -1, 24, -15, 24, 5, -17, 16, 1}
, {19, 3, -16, 24, -16, -2, 16, 1, 11, -2, 19, 13, -15, -15, 4, -11, 18, -13, 0, -7, -22, 18, 7, 5, 16, 1, -8, -16, 22, -1, 16, 4}
}
, {{25, 22, 20, 22, 2, 18, -27, -17, -21, 11, -15, 17, -5, -8, 4, 19, 22, 8, -26, -13, 14, -13, 17, 14, -15, -3, -11, 6, 17, 1, -1, -20}
, {21, -15, -19, -1, 25, 21, 6, 17, 26, -5, -14, -12, 18, 0, -22, 0, 3, -9, 22, -13, -8, 16, -9, 4, 18, 20, -20, 28, -15, -20, -23, 16}
, {-12, 25, 24, -4, -19, 5, -3, -8, 0, 17, -19, 13, 11, 14, 16, -20, 26, 13, -20, 15, 5, -19, -9, -19, -4, 17, -12, -22, -21, -22, 5, 10}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_271_H_
#define _BATCH_NORMALIZATION_271_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       53

typedef int16_t batch_normalization_271_output_type[53][16];

#if 0
void batch_normalization_271(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_271_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_271_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_271.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       53
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_271(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_271_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_271_bias[16] = {-33, -23, 66, -8, 95, 26, -28, -67, -29, 36, 14, 35, 52, 47, -56, -48}
;
const int16_t batch_normalization_271_kernel[16] = {156, 209, 180, 121, 143, 154, 181, 198, 141, 172, 161, 164, 118, 209, 111, 132}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_347_H_
#define _CONV1D_347_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       53
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_347_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_347(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_347_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_347.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       53
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_347(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    16
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_347_bias[CONV_FILTERS] = {-1, 0, -1, 0, -1, 0, -1, 0, -1, -1, -1, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0, 0, -1, -1, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1}
;

const int16_t  conv1d_347_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{12, 6, -9, 18, 3, 10, 12, 9, 13, 16, -15, -11, 19, -14, 19, -10}
, {-17, -8, 5, -17, 17, -13, -2, -20, -15, -22, -10, 1, 7, 19, 1, 17}
, {-13, 12, 6, -13, 8, -8, -9, 13, -16, 7, -4, 8, 3, 9, 9, 17}
}
, {{11, -12, -6, 8, 10, -12, 13, 2, 3, 17, -6, 5, 12, -9, -14, 2}
, {-18, 13, -16, 14, 18, -15, -18, -2, 3, -1, -6, -10, 6, -8, 17, 13}
, {19, -14, 13, 10, -14, 2, -1, 3, -14, 10, -15, 16, -2, 0, 6, 6}
}
, {{-10, 3, 12, -5, -1, 3, 2, -7, -19, -20, -13, -10, -20, -1, 4, -13}
, {-7, 16, 15, -13, -11, -8, -12, -10, 2, -12, 10, -2, -16, 11, 8, 13}
, {-3, 3, 19, 2, -15, -13, 19, -20, 7, 15, 1, 5, 19, 21, 8, 19}
}
, {{-5, -8, -17, -2, -20, -17, -7, -5, -10, -19, -9, 21, -1, -18, -7, -21}
, {-2, 2, -19, 13, 5, 17, -12, 17, -8, 2, 10, 2, -18, -5, -2, -5}
, {-19, 9, -6, -6, 6, -8, 13, 14, -13, 1, -21, -16, -18, 1, 7, 16}
}
, {{-6, -13, -13, -17, -9, 1, -13, 3, 12, -6, -12, 8, -15, 20, 13, -23}
, {-3, -8, -20, -16, 4, -12, -18, -14, -3, -14, 3, 8, 2, -11, -9, -12}
, {2, -5, 1, -13, 19, -20, -15, -18, -8, 3, 12, 0, 5, 10, 11, -10}
}
, {{9, 6, -8, 6, 3, 17, -20, -17, 10, 17, -11, -28, 9, 15, -11, -17}
, {16, 12, -10, 0, -17, -18, -16, -1, 0, -20, 11, -8, -18, 9, 16, 16}
, {16, -8, 12, -4, 17, -2, 20, -10, 22, -2, 16, -15, 7, -16, -6, 13}
}
, {{8, 11, -1, -5, -1, 0, 6, 1, -15, 4, 17, 20, -16, 18, 5, 9}
, {15, -2, -7, 3, -19, -14, -4, 12, -15, 2, -20, 4, 18, 0, 2, 13}
, {-17, 4, -2, -16, -11, -13, -1, 7, 10, 8, -4, 18, -7, 17, -21, 6}
}
, {{8, -11, 18, 7, -18, 22, -16, 2, 7, -16, 2, -10, 9, -3, 3, 7}
, {13, -6, -11, 19, -2, 15, 1, -17, 17, 20, 10, 2, -7, -5, 15, -13}
, {16, -10, 6, -13, 12, 4, 15, 21, -10, -5, 4, 14, 15, 11, -14, -20}
}
, {{15, -5, 13, 10, -14, 18, 4, -5, 14, -14, 10, 9, -3, 13, -5, -9}
, {-1, 1, -18, -3, -7, -15, 3, -6, 13, 12, 8, -10, 13, -6, 15, -8}
, {19, 3, 0, 5, -5, -13, 17, 7, -1, -8, -5, 23, 20, 11, 6, -12}
}
, {{-10, 4, 19, 2, -7, 18, 4, 1, -5, -11, 4, -15, -3, 12, 5, -7}
, {1, -20, 0, 0, -5, 3, 6, -15, -11, -1, -18, 16, -17, -8, -5, 16}
, {-19, -4, 2, -17, 6, -15, 11, -13, -10, 12, -11, -10, -11, -14, -1, -17}
}
, {{18, 7, -9, -6, 5, -18, -3, -7, -12, 1, 9, -4, -12, 0, -4, 17}
, {-10, 9, -10, 2, -12, 1, 5, 2, -6, -17, 17, 15, -10, 12, 14, 11}
, {11, 19, -8, 12, 17, -3, 13, -5, 3, -13, 2, 12, 13, 8, -17, -18}
}
, {{-16, 6, -16, -7, 12, 17, -6, 1, 3, -10, -21, -12, -9, -11, -3, -3}
, {-18, -9, -16, 1, 16, 20, 6, 4, 2, -6, 6, 8, 12, 12, -15, -8}
, {13, -3, -12, -19, -16, 1, 17, 16, -1, -9, -20, -1, -12, 19, 16, 14}
}
, {{-10, -11, -9, -13, -2, -18, -16, -2, 14, 15, -20, -3, -6, 0, -13, 3}
, {23, -16, -3, 3, 10, -13, 18, -13, 12, 16, 18, 4, 1, -9, 19, 6}
, {-16, -8, 14, 17, -11, -8, -11, -8, 15, -16, -7, -10, -17, 9, 16, -8}
}
, {{-5, -15, -10, 1, 13, -10, -3, -1, 14, -15, -20, 0, -13, -1, -6, -11}
, {-13, -3, -1, 12, 18, 7, 10, 11, 9, 3, -10, 11, -7, 10, -10, 11}
, {-16, 5, -7, -7, 16, 12, 5, -14, -19, -1, 1, 5, 3, -4, -21, 5}
}
, {{-15, -19, 14, 4, 16, -20, 16, -6, -15, -15, -6, 14, 17, 6, 15, 4}
, {-10, -5, -20, 3, 4, 17, 13, 10, 16, -2, 6, 11, 4, -2, 12, 2}
, {-3, -16, -15, -2, -15, -11, 12, -22, 15, 7, -17, 14, 20, -5, 21, 5}
}
, {{11, -7, 19, 8, 18, 12, -11, -15, -21, 21, -17, 1, 10, -14, -2, -3}
, {15, -1, -15, 14, 6, 22, -6, 15, -21, -16, -12, 9, 16, -7, 14, 14}
, {-6, 6, 5, -17, -1, -7, 3, 10, -1, 18, 4, 0, -13, 9, 13, 0}
}
, {{10, 7, 6, 2, 16, -7, -7, 5, -4, -10, 9, -2, 7, -5, 14, 15}
, {-19, 16, -18, 19, 12, 18, -16, 13, 13, -15, 22, 8, 6, 1, 5, 1}
, {-6, 20, -20, -18, 9, 4, -15, 8, 2, 19, -2, 16, 12, -9, -1, 10}
}
, {{-9, 0, 4, 14, -9, -20, -19, -18, 6, -1, -3, -15, 8, 15, -15, -11}
, {10, 10, 7, -12, -7, 19, 11, 16, 7, 3, 22, -10, 11, 8, 8, -2}
, {14, -17, -15, 2, -12, -1, -17, -2, -15, -13, -11, 11, -23, 3, -12, 9}
}
, {{7, -21, -19, -4, 14, -18, 8, -3, 7, -17, -16, -4, -2, -9, -3, -3}
, {-15, -18, -3, 4, -6, 9, 0, 0, 0, 4, -17, 3, -5, 14, 11, 16}
, {3, 4, 18, -16, -13, 0, -9, -14, -2, -3, -14, -16, 3, 1, -9, 19}
}
, {{-12, -10, 16, 13, 18, 13, -18, -10, -10, 15, 10, -18, 14, -8, -11, 17}
, {10, 13, 2, -15, -14, -18, 22, -7, -2, -1, -15, -4, 8, -10, 7, -6}
, {-11, 23, -24, 13, 16, -2, -11, -5, 4, 17, -6, 14, -3, 9, -20, -8}
}
, {{15, -2, -7, 15, -14, -8, 14, 17, -15, -11, -2, -7, 7, -12, -21, 1}
, {-1, -2, -14, -5, 21, -9, -7, 3, 12, -5, -14, 12, -8, 5, -17, -14}
, {-13, 4, -12, -3, 2, 4, 4, -10, -6, 6, 9, -22, -9, 13, 5, -19}
}
, {{-2, -16, 13, 21, -17, 5, 14, -4, 18, -18, 19, -14, 13, -15, -16, -6}
, {16, 8, 15, 15, 14, -21, 4, 2, -14, -3, -21, 6, 9, -15, 20, 17}
, {-18, 4, 19, 16, 19, -9, 10, 14, -17, -9, -1, -9, 10, -4, -16, 0}
}
, {{-2, -3, 9, 16, 11, 9, -3, 21, 14, -21, 3, 9, -14, 13, -2, -9}
, {19, -7, -13, -1, 8, 2, 0, -7, 10, -12, -15, 9, 11, -13, 4, -19}
, {11, -8, -6, -19, 2, 9, -16, -8, 13, -11, 13, 12, 12, -16, 3, -4}
}
, {{-19, -17, 13, 2, -17, -4, -12, 0, -14, 13, 12, -22, 13, 1, -14, -13}
, {-18, 10, -10, -9, 0, 3, -14, -20, -14, -17, 17, 9, 16, 15, -21, 16}
, {17, 6, -10, -13, -11, 8, -18, -13, 4, -11, 18, 3, 0, 14, 15, 14}
}
, {{-3, 14, -1, -18, 5, 16, -19, -10, 4, -15, -17, 12, 10, -6, 13, -5}
, {17, 3, 7, -1, 14, 16, -16, -7, -3, -2, -14, 16, -19, 5, -1, -16}
, {-14, -21, -3, 3, 0, 1, 2, -9, -4, 3, 2, -1, -12, 1, 14, 6}
}
, {{-17, -11, 2, -20, -12, 5, -5, 13, -19, -5, -19, 8, 13, 13, 11, -12}
, {11, -19, 12, 7, -10, -11, -3, 2, 4, 17, -8, 6, -15, 7, -5, -4}
, {20, -5, 2, 10, 3, -6, -16, -10, -12, 0, -4, -14, 10, -3, -18, 10}
}
, {{6, -18, 15, -8, -10, -11, -8, 8, 17, 7, 11, -18, -8, 8, -16, -18}
, {-4, 22, 12, 2, 18, 17, 20, -17, 5, -20, 4, -13, -17, -12, 8, 20}
, {-11, -13, -10, 10, 9, -10, -5, -5, 15, -2, -7, 8, -16, 12, -18, -10}
}
, {{-14, 8, -7, -13, 16, 19, -15, -18, -14, -7, 1, 7, 7, -3, 1, -2}
, {4, -8, -13, 5, 7, -4, 5, -25, -7, -13, -12, 13, 2, -14, 18, -2}
, {-9, -2, -8, 5, 17, -14, -8, -18, -12, -6, -20, 5, -10, 2, 1, 9}
}
, {{23, 2, -14, 16, 11, 8, -14, 0, -4, 21, 0, 7, -1, 9, -14, -16}
, {8, 17, -2, -20, -10, 6, 0, -14, 6, -10, 9, -9, 18, -3, -1, 11}
, {9, -15, 11, 8, -8, -8, -6, 16, -7, 6, -4, 15, 15, -2, -10, -18}
}
, {{2, -5, 9, 9, -14, 1, 14, 7, 6, 9, 0, 11, 4, -22, -7, 7}
, {-20, 11, -10, -11, -12, 13, -13, -20, -2, 15, 7, 5, 15, 5, -22, 10}
, {-21, -6, 0, -4, 12, 3, -15, -5, -12, -6, 10, 23, -6, -21, -15, -11}
}
, {{15, -17, 18, -4, -13, 16, -20, -17, 5, -16, -5, 0, -1, -17, -18, -7}
, {20, 18, -7, 17, 1, -7, -10, 7, -13, 11, 12, 4, -8, 21, -11, 7}
, {-15, -16, -3, -4, -11, 4, -19, 5, 2, -8, 19, -9, -2, -20, -16, -10}
}
, {{-10, -10, 24, 11, -4, -18, -15, -10, 16, 9, 2, 0, -19, -8, 6, -10}
, {-13, 4, 0, 16, 2, 3, 3, -8, 10, 14, -21, -17, 6, -8, 14, -12}
, {-7, -21, 10, 4, -1, -4, 6, 17, 9, -5, -10, -3, 20, -20, 7, 18}
}
, {{10, -13, 3, -1, -2, 13, -19, -10, 5, -4, -19, 20, 1, -16, -9, 4}
, {10, -6, -12, 11, 3, 7, 0, -4, -21, 8, -16, 16, -4, -10, -12, 5}
, {13, 19, -16, -10, -14, 10, -11, 8, 11, -6, 0, 13, 18, -4, -17, -8}
}
, {{3, 13, -19, -1, 25, 17, 5, -17, -20, 12, 4, -7, -14, 3, 5, 10}
, {-15, 13, -16, 2, -11, -10, -4, 2, -8, 19, -9, -11, 13, 12, -9, 20}
, {12, 20, -15, -9, 0, 8, 10, -8, 16, 9, -19, -18, 9, 19, 15, 17}
}
, {{-12, -9, 6, 17, 4, -7, 15, 11, -6, 7, 17, -4, -7, -16, -20, 1}
, {-16, 7, -13, 16, 6, 8, -6, 16, 11, -8, 1, 1, -17, 7, 5, -7}
, {-14, -22, 20, -11, 16, 18, -3, 7, -4, 6, 3, -10, -22, -16, 17, -1}
}
, {{12, 7, -10, 15, -6, -6, 12, 10, -7, -5, -20, 8, -14, -18, -9, -3}
, {-14, 3, -9, 14, 11, -17, 1, -14, 15, -19, 6, -8, -4, -3, 12, 15}
, {16, -16, 8, -21, -16, 4, -1, 18, -7, 15, -12, 13, 8, 11, 11, 5}
}
, {{6, -5, 9, 2, -8, 19, -15, -14, -21, -12, -4, 0, 13, -13, 8, -5}
, {20, 20, -7, 3, 4, -11, 6, -8, -3, -6, 9, 4, -7, 12, -19, -1}
, {-10, 1, 4, 10, -20, 8, -16, -11, -9, -8, -7, -1, -12, 19, -21, -21}
}
, {{-12, 16, 11, -3, 19, 1, -15, -12, 14, -3, 1, 11, -22, -13, -10, -10}
, {-15, 8, -8, 13, -10, -4, 1, 8, 9, -17, 1, 1, -2, -2, 8, 2}
, {-17, -9, -13, -6, 7, 11, 2, 7, -14, -17, -5, 4, -18, 23, -3, 1}
}
, {{8, -23, 18, -9, -1, 12, 10, -8, -8, -28, 12, 13, -18, -14, -7, 20}
, {-22, -4, 10, -11, -7, -15, 10, -14, 11, 7, -20, -1, 18, 12, 5, 7}
, {-21, -23, -15, -2, 10, -2, -12, -13, -3, -18, 5, -1, 3, -17, 1, -22}
}
, {{-15, 11, 1, -3, -15, -14, 18, -8, 18, 17, -3, -8, -13, 6, 4, -8}
, {17, 8, -14, -1, 9, -21, 18, -3, 10, 11, 0, -6, 4, 22, 1, -11}
, {11, -5, 1, -10, -2, -20, 1, 3, 12, -4, 17, -20, 6, -15, 0, -7}
}
, {{14, -22, 0, 9, -19, 6, -15, -9, -9, 18, 20, 15, 9, -8, 2, -10}
, {-11, 8, -6, -16, -7, 1, -6, -15, 15, 5, 3, 2, -4, 9, 10, 4}
, {-9, -6, 17, 9, 10, -19, -3, 10, 11, -18, 3, -6, 17, -11, -2, -21}
}
, {{12, 1, -5, -11, -7, -17, 14, -8, 18, -7, -14, 2, -16, -10, -8, 11}
, {19, 15, -15, -12, 3, -3, -1, 10, 16, -8, -15, 11, -18, 10, 18, -14}
, {13, 11, -14, 6, -8, 15, 12, -6, -9, 9, -6, 14, -18, -8, -3, -16}
}
, {{5, 4, -20, 15, 3, -8, -14, -16, 13, -20, 2, -3, 12, 7, -8, -16}
, {13, -12, -22, -20, 16, -11, -17, 2, -21, -1, -17, 17, 1, 3, -17, -13}
, {4, 18, 9, -4, 10, -9, 18, -6, 15, 10, 12, -20, -12, 14, -14, 16}
}
, {{-12, 1, 16, 13, 8, -17, -9, -5, 0, -11, -16, -5, -6, 4, -1, -7}
, {-12, -3, 12, 15, 12, -7, -4, -19, -14, -9, 14, 2, 12, 17, 1, -7}
, {8, 10, -4, -9, -17, -1, 20, 8, 1, 0, 18, -13, 5, -9, -16, 17}
}
, {{17, 1, 9, -8, -13, 1, 20, 20, -20, -6, -7, 23, -16, -19, 4, 11}
, {-5, 1, 1, -8, -13, -2, 12, -11, 1, -5, -2, 10, -15, -2, -16, -11}
, {7, 11, 19, 10, -7, 1, 0, 8, -10, 14, -10, 15, -9, 4, 20, -13}
}
, {{-5, 10, -7, -15, 18, 4, 19, -18, -10, -12, -9, 13, 6, -16, 4, 2}
, {3, 1, -10, 6, 14, -18, -6, -18, 13, 2, -3, -12, -16, 2, -10, -18}
, {-21, 16, 19, 18, 3, -8, -14, -1, 7, 12, 1, -11, 11, 3, 8, -2}
}
, {{-8, -4, 6, -14, -8, -11, -13, -8, -12, 2, -12, 9, 1, -14, 10, -6}
, {-6, -11, -13, -19, -20, 9, 12, -16, -2, -7, -18, -16, 13, -14, 9, -8}
, {6, 4, 6, -19, 2, -21, 12, -8, 5, -7, -4, -1, -1, 18, -5, 17}
}
, {{-13, 2, -21, -4, 6, 5, 1, -11, 14, 18, 13, -10, 16, 2, -12, -18}
, {19, 12, -21, 10, 10, -1, 6, 3, 14, -8, -6, -9, 13, 2, -7, 6}
, {8, 16, -1, -5, 17, 16, 6, 0, 15, -1, -1, 0, 2, -5, 16, -16}
}
, {{4, 6, 9, 15, -10, -4, 6, 17, -15, -15, 6, -13, 13, 17, 18, -15}
, {0, -19, -3, -3, -1, -16, 4, 11, 10, 15, -9, 7, -21, 8, -4, -11}
, {-1, 14, 4, -17, 11, 7, 1, 10, -3, -16, -17, 7, 5, 12, 16, 0}
}
, {{-1, -9, 12, -1, 12, 16, 9, -20, 10, 1, 5, -12, 10, 18, 5, -2}
, {-10, 21, -4, 19, 10, 0, -3, -1, -16, 3, 18, -12, -3, 1, 16, 3}
, {4, -3, -6, 18, 4, -6, -8, 1, 17, -11, 8, 21, -18, -7, -10, -2}
}
, {{15, 9, -4, 10, -15, 1, -5, -10, -18, 18, 1, -6, -8, -5, 19, 16}
, {-19, 12, 5, -17, -3, -2, 5, -8, -6, 7, 0, -8, -4, -12, -10, 10}
, {7, 1, -17, 9, 24, -9, 1, -11, 0, -5, 22, 3, 20, 18, -1, -14}
}
, {{9, -13, 6, 5, 21, -8, -2, 15, 1, -10, -9, -4, -21, 13, 5, -3}
, {-8, 8, 2, 15, 6, 5, 16, 16, -11, 18, 8, -18, -5, -7, -5, 18}
, {-4, 19, -10, -7, -1, 3, -2, -6, 15, 3, 8, -16, -18, 4, -20, 4}
}
, {{-16, -13, -6, 5, 6, 18, 19, 10, 16, -22, -2, -1, -18, -1, 10, -8}
, {-5, -19, -18, 14, -24, -15, -12, -8, -13, 14, 4, 17, 10, -21, -20, 5}
, {19, 21, -11, 0, 13, 12, -2, -13, 1, 10, 12, -15, 3, -8, -11, -16}
}
, {{-1, 1, -10, 16, -6, 4, 1, 8, 3, 7, -9, -14, -3, -15, -7, 5}
, {4, -10, 12, 11, 11, -3, 11, -11, 8, 11, -8, 2, -4, 17, 16, 15}
, {10, -12, -16, -4, -16, 18, 10, -2, 14, -14, -15, 16, -19, -3, -6, 14}
}
, {{11, -3, -6, -20, 10, -7, -9, 10, 6, 7, 4, 0, 16, 19, 16, 21}
, {9, -8, -7, 10, 0, -9, -4, 8, -11, -14, 9, 6, 11, 8, -7, 11}
, {-16, -18, 2, 0, 15, 18, 12, -13, 3, -10, 9, 11, 1, 15, -16, -17}
}
, {{-14, 2, -17, 2, -16, 8, -12, -4, 4, -6, -5, -14, 7, 22, 11, 12}
, {-1, -8, -17, -20, 14, 1, -16, -9, 15, -2, 20, 15, 1, 6, -19, 3}
, {-14, 5, 4, -11, -4, -17, -5, -3, 5, -19, -7, -11, -17, 21, 6, -13}
}
, {{8, -15, 13, -3, 10, 19, -5, 8, 1, 15, 1, 9, 0, 3, 15, -14}
, {11, 10, 6, 17, 8, 9, -7, -18, -10, -15, -21, -1, -16, -19, 5, -12}
, {0, 17, 18, -15, 1, 20, 7, -9, -7, 8, -15, 19, 0, 4, -8, -1}
}
, {{13, 12, 3, -1, 5, 7, 16, -16, -4, -15, -4, -8, 3, -4, -3, 5}
, {5, 11, -8, -9, -22, 4, 8, 0, 18, -6, -8, -11, 0, -14, 18, -19}
, {-9, 9, -5, 8, 10, 3, -7, -17, 11, -18, -15, -11, 0, 15, -8, -3}
}
, {{5, 22, -23, -11, -19, 1, 4, 2, 14, 17, 1, 8, -13, -17, -22, -11}
, {9, 16, 8, -3, -2, 13, 19, 7, 5, 15, 17, 11, -8, -8, -9, 16}
, {-2, -15, 1, 12, 2, 13, 20, 11, -19, -16, -20, -16, -18, -11, -3, 13}
}
, {{17, -5, -3, 1, 15, 7, 18, 4, 3, -19, 0, -15, -14, 12, -22, 18}
, {-8, 7, 4, 15, 15, 11, 3, 13, 1, -17, 6, 12, -16, 14, 7, -18}
, {9, 12, -17, 10, 7, 8, -14, 0, 17, 4, -8, 2, 12, -7, -14, -21}
}
, {{13, -14, -8, -12, -10, 19, -5, -1, -23, -10, -18, -19, 1, 18, -10, -1}
, {11, 6, 2, -4, 1, -3, 13, -6, -1, -1, -2, -6, 12, 12, -18, -13}
, {5, 3, -11, 19, -16, 7, 6, 7, -17, -17, -13, 13, 10, 16, 19, 11}
}
, {{19, 4, -5, -7, 18, 11, -22, 15, 11, -7, -17, 7, -15, 15, 14, 14}
, {1, 1, 2, 1, 5, 5, -13, -7, 17, -5, -8, -6, -8, -18, 0, -3}
, {8, 19, 5, 2, 7, 7, 12, -10, -10, 14, 4, 17, -10, 5, 12, 18}
}
, {{-8, 15, 7, -13, -14, -14, -1, 18, -11, -20, -14, 12, 12, -1, 0, -7}
, {-14, -2, 2, 0, 13, -2, 13, -7, 4, 11, 16, 0, 9, 23, 18, 9}
, {18, -16, -17, 1, -1, -14, -14, -6, -8, -17, 14, 16, -14, 22, 18, 18}
}
, {{-18, -12, 15, -6, 10, -1, 12, -8, 10, -12, -19, 2, -11, 0, -5, 0}
, {0, -7, 5, 13, -4, 17, 8, 16, -15, 11, -9, 2, -16, 4, -2, -2}
, {-17, 3, -7, -15, 1, 5, -7, -7, -11, -15, 7, 11, 2, 20, -5, -16}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_272_H_
#define _BATCH_NORMALIZATION_272_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       51

typedef int16_t batch_normalization_272_output_type[51][64];

#if 0
void batch_normalization_272(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_272_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_272_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_272.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       51
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_272(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_272_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_272_bias[64] = {-36, -31, -9, 90, 79, -17, -44, -92, -85, 68, -51, 26, 16, 32, -34, -54, -84, 35, 89, -3, 58, -40, -21, 37, 19, 38, 10, 70, -41, 14, 50, 6, 18, -45, -20, -11, 40, 34, 68, -17, 2, -2, 29, -1, -37, 16, 82, -73, -40, -58, -29, -34, 29, -34, -70, 66, -35, 32, -30, -61, 10, -72, -41, 23}
;
const int16_t batch_normalization_272_kernel[64] = {260, 270, 268, 281, 189, 258, 341, 319, 282, 264, 315, 317, 246, 334, 223, 287, 236, 319, 363, 309, 274, 276, 338, 288, 325, 275, 320, 257, 287, 231, 246, 250, 222, 203, 300, 279, 233, 335, 217, 288, 347, 345, 243, 355, 373, 350, 244, 294, 321, 244, 345, 251, 277, 232, 352, 325, 263, 352, 273, 297, 276, 238, 234, 295}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_348_H_
#define _CONV1D_348_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       51
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_348_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_348(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_348_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_348.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       51
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_348(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    64
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_348_bias[CONV_FILTERS] = {-1, 0, 0, 0, 0, -1, 0, 0, -1, -1, 0, -1, 0, 0, 0, -1, 0, 0, -1, -1, 0, -1, -1, 0, 0, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, -1, 0, -1, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, -1, 0, 0, -1, -1, -1, -1, -1, 0}
;

const int16_t  conv1d_348_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-13, -2, -4, -1, 3, 14, -13, 7, -4, -4, 0, -12, 12, 12, -14, -10, 14, -2, 6, 16, -10, 8, -1, -10, -15, -7, -4, 11, -1, 1, -1, 7, -3, -3, 2, 7, -15, 4, 3, 18, 4, 3, 6, -3, -15, -17, 1, -5, 11, 1, 7, 16, -5, 1, -4, 5, 9, 0, -6, 16, -12, -4, -7, -13}
, {-4, -3, 8, -8, 16, -4, 9, 2, 5, 1, -13, 8, -17, 13, 2, 6, 13, -12, 11, 13, 11, 10, 12, -5, 7, -8, -3, -14, 3, -6, 4, 7, -11, -2, -1, 5, 7, 4, -6, 19, -14, -15, -3, 7, 6, -15, -15, 14, 9, 4, 12, -2, 2, -9, -9, 9, 4, 2, -7, -11, -4, -6, -7, 14}
, {-5, -3, 13, 2, -5, -13, -10, 1, -3, 9, -15, -9, 9, -11, -3, 9, -15, 14, 11, -12, 3, 7, -15, -12, 9, -16, 7, 12, -20, -14, -14, 2, 1, -11, 1, -5, 0, 1, 15, 17, 9, -14, -16, -9, -7, -15, 11, 3, 14, 9, -1, 13, 2, -10, 0, 3, 12, 12, 9, -9, 15, 12, 11, -6}
}
, {{3, -14, -3, 10, 11, -14, -3, -14, 3, -2, 6, 14, 3, 4, -14, -17, -14, -5, 14, 5, 12, 10, 7, 0, 9, -13, -13, 1, -18, 5, 6, -10, 1, -7, 2, 6, 5, -9, -14, 9, -12, -12, 0, 6, -5, -18, 9, -8, -11, 5, 15, -3, -17, 2, 16, 18, 7, -13, -9, 6, -4, 9, -8, -3}
, {-9, 4, -2, -6, -11, 9, 0, 13, -4, -14, -7, -10, -10, 11, 15, 0, 10, 11, 0, 10, 12, 1, -3, -5, -6, 7, -10, 7, -7, -10, -13, -3, -14, 10, 12, -11, -3, 6, -12, 16, 14, -10, 5, 19, -5, 14, 11, -9, -10, 2, 2, -7, -15, 1, -5, 11, 0, 2, -9, 12, -10, 1, 0, -2}
, {-6, -5, -14, 11, -3, 6, 16, 8, -14, 9, 6, 1, -14, -10, 1, 8, 7, 19, -4, -17, -5, -13, -3, 2, 12, -5, 15, 4, 0, -8, -12, 9, -11, 6, 5, 13, -4, -2, 4, 11, -3, 8, -2, 5, 3, -3, 12, 5, 7, -12, 11, 11, -14, 5, 2, -9, -15, -3, -13, -3, -9, 13, 4, 9}
}
, {{13, 15, 10, 5, 3, -2, 10, 3, -10, -4, -13, -7, -10, -2, -11, 2, -5, 13, -8, -1, -15, 14, 11, 0, -17, -7, 10, -1, -8, 6, 12, -5, 5, 10, -1, 7, 0, 12, -16, -3, -1, 3, 9, 1, 13, -3, -2, 3, -7, 16, -3, 11, 0, 8, -7, 0, 7, -4, 0, -1, 4, -11, 4, 10}
, {12, 0, -2, 6, 5, -13, 18, -14, -3, 13, 6, 3, -13, 6, -7, -2, -8, -8, 16, 4, 16, -3, -18, -8, 10, -2, 5, 13, -10, 14, 5, -12, 4, -14, 12, 6, 4, -7, 8, -7, -9, -8, 5, 14, -12, -10, -9, -15, 9, 11, -6, -2, -1, -8, 3, 9, 2, -12, 17, -1, -7, -9, -5, 3}
, {5, 0, -12, 10, -1, 9, -10, -5, -14, -9, -14, -1, -6, -8, 3, -15, -10, -1, 15, -13, 9, -10, 13, 15, 10, 0, 12, -4, -9, -3, 3, 1, -9, 3, 3, -11, -13, -10, 3, 9, 5, 9, 13, -7, -15, -11, -7, 2, 14, -5, -12, -2, -18, -14, -7, 6, 6, -3, -12, 17, 6, -6, 5, -2}
}
, {{-6, -4, -6, -9, -9, 12, -1, -6, 9, -14, 6, 9, -4, -1, 9, 12, 12, 11, -3, 11, -3, -16, -2, 5, -16, 9, -13, 1, -3, -4, 3, -1, -9, -2, 2, -6, 5, 5, 13, 7, -17, -15, -12, -14, -4, -3, -15, -10, 8, -15, 14, -4, -7, -2, 0, -4, -11, -16, -7, 15, 2, 10, 14, -2}
, {-3, -5, -10, -4, 1, -15, -10, -10, 4, 10, -7, -11, 4, -8, -16, -14, -2, 2, -1, 3, -2, 9, 12, 9, -4, -11, -9, -16, 2, 10, 11, -16, -10, -7, -2, 15, -8, -12, -20, 12, 7, 3, 11, 12, -9, -4, -9, 0, -6, 14, 3, -2, -7, -12, -7, 13, -11, 8, 0, -16, 14, -5, -4, -7}
, {-8, -8, 14, 2, 2, -8, 13, 9, -1, 8, -6, -8, -12, 11, 3, 11, 16, -8, -14, -1, -5, -11, 16, -6, -11, 3, 17, 8, -6, 3, -9, 3, -19, 7, -2, 9, -6, 18, 8, -10, -11, -18, -14, -2, 0, -14, -16, 7, -13, -10, 8, -5, 10, 3, 13, 4, 3, 1, -14, 1, 6, -8, -9, 1}
}
, {{-9, -7, -2, -2, 9, -4, -11, 9, 7, 4, -4, 7, 1, -11, -4, 3, -6, -8, 9, -13, -14, 11, -9, 11, 5, -11, 7, 13, 8, 4, 6, 0, -8, 3, 5, -4, -10, 8, 11, 14, 15, -1, -1, 20, 6, -1, 0, 0, -5, 0, 12, -3, 4, 1, -11, 1, 12, -10, -11, 17, -7, -15, -10, -1}
, {2, 15, -7, -12, 7, -13, 0, 16, -9, 8, -16, -4, 3, 3, -7, -6, 12, -2, 1, -14, -5, -7, -4, 0, 8, -2, 7, 7, -9, -11, -11, -4, 7, 1, -6, 1, -10, -3, 12, 14, 10, 5, -11, 2, 12, -2, -9, -12, -2, 6, 8, -5, 12, -12, 4, -11, 10, 4, -5, -14, 8, 6, -12, -4}
, {-1, -14, 13, -5, -5, 8, 5, 2, 7, -14, 3, -6, 9, -8, 16, 11, -15, 13, 13, 5, -1, 6, 3, 0, -3, 1, 0, 5, 1, -8, 11, 6, 13, -11, -1, 4, -16, 13, 2, 11, 6, -13, 0, 7, -4, 8, 1, 13, 15, 15, -8, 1, -9, 13, -14, -1, -10, -11, -14, -6, 0, -16, 15, 10}
}
, {{11, -16, -6, 13, -2, 10, -6, 9, -10, -10, 13, -6, -17, 3, -13, 12, -15, -5, 12, 1, 8, -7, -14, 13, 16, 8, 3, 16, 7, 8, 8, 9, -12, -8, 3, -1, -4, 20, -14, -2, -1, 6, 2, -12, 14, 2, 14, -5, 14, 5, 12, 13, -4, 4, 8, 4, 11, -12, 16, 15, 16, 4, -10, 10}
, {-13, -16, 15, -6, 9, -13, -6, -11, 9, -8, 5, -4, 2, 6, -8, -16, -9, 7, -3, -6, 0, -6, 19, -6, 10, 13, 7, 16, 8, 6, 11, 14, -7, 11, -13, 1, -10, 17, 12, -8, 1, 14, -2, 14, -12, -5, 3, -11, -9, 9, -2, -8, -3, -9, -10, -2, -9, -4, 2, 0, 8, 17, 16, -5}
, {-6, 10, 0, 0, -4, -13, 11, -14, 4, 5, -10, -5, -10, 7, 13, 12, 14, -13, -8, 1, 1, 9, 15, 5, 2, -10, -12, 1, -4, -10, -14, 11, 0, 9, -8, 5, 11, 11, 5, -1, -3, 16, 9, -10, 7, 4, 6, -5, 1, 12, 0, 9, 2, -2, 2, 4, 8, 9, 8, 7, 12, 17, 15, 9}
}
, {{12, 7, 2, -1, -2, 9, 7, 4, -6, 8, 7, 3, 12, -3, -8, -5, 9, -15, -7, -15, 13, 4, 12, -8, 7, -1, 4, 2, 1, 12, 18, 14, 18, 0, 0, 14, 6, -1, 13, 6, 1, -12, -1, 12, -5, -5, 0, 14, -12, -9, -11, -4, -5, -6, -9, 6, 13, -2, -5, -16, -10, -9, -12, -3}
, {3, -13, -6, 8, -3, 16, 9, -4, 5, 3, 7, -17, 14, 1, -5, 11, -3, 2, -15, -6, -5, -6, -13, 16, -16, 10, -15, -15, 15, 9, 11, -14, 8, -7, 4, -9, 10, 10, 10, -15, 7, -3, -11, -7, 10, 1, 9, -4, 8, 15, 10, 4, -16, -11, -15, -13, 1, -9, 13, -15, -4, -12, -11, 1}
, {-17, -5, 8, 10, 12, 9, -5, 7, -17, -12, 10, -1, 15, 10, 0, 12, -4, -1, -18, -12, 10, -9, 2, -1, 13, 16, 1, -11, -3, -15, -11, 1, 7, -11, 3, 8, -3, 3, 8, -8, 4, 3, 8, -4, -10, 5, -7, -4, 10, -11, 5, 14, 5, -13, -11, 14, 11, -14, -18, 12, -8, -5, 9, -10}
}
, {{14, 14, -9, -3, -12, -11, 0, 6, 16, -4, 14, -3, -7, -11, 11, 13, -8, -5, 3, -6, 4, -3, -7, 12, -4, 10, -18, -10, -6, -7, -10, -2, -5, -11, 1, 15, 3, -4, -10, 2, -8, 8, 16, 2, -8, 9, 5, 7, 11, -3, 0, 3, -12, -3, 8, 3, -13, 9, 6, -11, 2, 1, -6, -15}
, {12, -5, -5, -2, -4, -1, 12, 0, -15, 8, 2, -3, 10, 6, 1, 6, -4, 0, 8, -12, 4, -1, 13, 5, -2, -10, 15, -9, 9, 9, 15, 0, -11, 13, -6, 17, 11, -6, 0, 4, 13, -10, -4, -1, 2, -12, -1, 13, 4, 15, -6, -8, -7, 14, -1, 3, 7, -16, 3, -12, -13, -13, -3, -4}
, {6, -16, 14, 11, -1, -13, 0, 6, 15, 16, 15, -11, 9, -3, -4, -17, -15, -7, -16, -18, 6, -2, -13, 2, 6, 8, -3, 11, 6, -17, 9, -10, 6, 10, -17, -2, 6, 6, 11, 7, 10, 15, 5, -14, -12, -9, -4, 3, 3, -10, -6, -17, -14, -13, 9, -12, 1, -12, -7, 0, -11, -1, -11, 5}
}
, {{-12, -16, 2, 9, -10, -11, -12, 12, 2, 13, -15, -14, 11, 2, -19, -3, -15, 1, -12, 3, 5, -8, 4, 7, -1, 5, 16, 15, -4, 0, 3, -17, 10, 12, -2, 0, 4, 11, 2, -9, -4, 2, 0, 7, -18, -12, -12, -15, 15, -7, 7, -14, 11, 4, -14, 6, 8, 16, 5, -9, 7, 11, -12, -8}
, {8, -1, -13, -16, 3, -13, -8, 7, 2, -9, 5, -4, 1, 9, -4, 13, 4, 12, -16, 3, -4, -17, -12, -9, 4, 4, 1, -10, -2, 3, -15, -5, -19, -4, 4, -2, -4, 3, -8, 3, -3, 9, 11, -2, -5, 6, 5, 3, 4, 8, 3, 13, -14, 11, -7, -11, 3, -3, -4, 7, 12, -5, -5, -8}
, {-5, 9, -10, 8, 8, -5, -13, 2, -10, 18, -14, -8, -15, -1, 0, -8, -4, -13, 7, -7, -1, 2, 2, 14, 9, 2, -10, 6, -9, -3, 2, -4, 8, -9, 7, -11, 10, 11, 11, -11, 15, 6, 1, 14, -8, -14, -2, 2, 7, 1, -2, 6, 5, 12, -14, 8, 12, 1, 4, 1, 7, 6, 6, 12}
}
, {{-9, -7, -9, 1, -8, 0, -4, -10, 10, 0, -13, 9, 1, 7, 4, -11, 4, 0, 3, 5, 11, 3, 5, 9, 5, -8, -12, 14, 9, 12, -15, -14, 1, 1, 14, -10, 8, 8, -5, 9, -11, -10, -11, 5, -14, 0, 0, 14, 7, -7, 1, 5, -4, -4, 12, -11, -1, 13, -4, -12, 16, 12, 3, -8}
, {2, -17, 8, -10, -5, 10, -7, 9, 10, 12, -16, 0, -4, 1, 14, 16, 9, -5, 9, 16, 9, -13, 2, 0, -13, -11, 9, 14, -1, 7, -13, -16, 11, 9, -11, -8, 2, 6, 7, 3, -7, -10, -2, -8, -12, -1, 6, -2, -7, -9, 2, -7, -17, 10, -11, -4, 8, 15, -3, -4, -3, -1, -16, -3}
, {2, -8, -15, -2, 12, -6, 12, 2, 14, -1, -14, -17, -14, 15, 2, 16, -10, -12, 17, 0, -8, -1, -11, -15, 16, 4, -13, 14, -14, 9, -13, -15, -7, 3, -3, -2, -2, -14, -6, -1, 12, 2, -4, -10, -5, -3, 3, 10, 0, 5, -1, 0, -18, 0, -5, -1, 3, -4, -3, -6, 12, -4, 2, 10}
}
, {{12, 12, 11, -5, -10, 12, 1, -7, -5, 6, 4, 13, 14, 6, 9, -2, -18, -11, -13, -9, 7, -7, 12, -9, 6, 11, -5, 6, -9, 8, -16, 2, -9, 5, 8, -4, 8, -8, 14, -4, 3, -7, 5, -2, 7, -13, 1, -10, -12, 2, -15, 9, -3, -15, -4, 7, 7, -8, -7, 14, 16, -3, -15, -7}
, {9, 4, 5, 10, 12, -13, 6, -2, -1, 18, 13, 3, 12, -15, 3, 6, -9, -10, 0, 8, -1, 8, 10, -5, -10, 12, 9, -2, -18, 6, -11, -10, 12, 6, 5, 10, 5, -9, 10, -13, -12, 1, -15, 9, -16, 13, -9, 16, 9, -9, 1, -10, -1, -13, 13, 5, -18, 14, -2, 7, 16, 10, -6, -11}
, {10, -8, 15, -7, -7, -13, -7, 2, -7, 6, -4, -15, 1, -12, 7, -17, 15, -12, 3, -4, 12, 15, 6, 12, -12, -13, -15, -11, -6, 8, -7, -16, -12, 3, -5, 3, 4, -7, 6, 6, 16, -16, -11, 10, -10, -2, 13, -2, -9, -1, 5, -1, 4, 0, -15, 3, -3, 5, -9, 12, -3, -13, -18, -12}
}
, {{6, -3, -9, 16, 10, 14, -16, 14, 8, -16, 16, 4, -8, -6, 2, -4, 3, -11, -2, 16, 14, 7, 13, 13, -4, -8, -4, 13, -15, -5, -4, -6, 3, -4, 0, 7, 11, -5, 2, 7, -9, -11, -4, -11, 13, -8, 1, 3, 8, 13, -4, 4, -3, -1, -14, 9, 6, 13, 5, 0, -10, -2, -11, 0}
, {-8, 6, -1, -8, 16, -15, -7, -14, 5, -7, 3, 14, 10, 13, 0, -11, -16, -2, -18, 15, -2, -7, -7, 9, -11, -9, -3, 8, -14, -11, 8, 8, 14, 2, 14, -16, 14, 17, 7, 0, 2, 11, -13, -1, 11, 8, -2, 16, 13, 15, 10, 3, 13, 3, 1, 8, 2, 3, 4, 15, -16, -17, 9, -15}
, {11, -11, -11, 8, 9, -1, -4, 5, 7, 2, -4, 13, -5, 0, -8, 4, 2, -5, 4, -7, 6, -8, -3, 0, -11, -6, -5, 7, 0, 13, -11, -16, 4, 5, -13, 2, 14, 0, 11, 14, 8, 7, -13, -3, -11, 9, 8, -13, 16, 7, -17, 0, -5, 4, 10, 7, 15, -2, -8, -3, 12, -13, 7, -8}
}
, {{7, 1, -12, -1, -1, -15, -12, 0, -8, -6, 0, 16, 13, -3, 7, 6, -4, 13, 10, -13, 7, 14, 8, 10, 2, 14, -13, -7, 3, -6, -2, 1, 9, 2, -15, -15, -8, 15, 8, 2, -11, 3, 12, -11, -11, -10, 7, -11, -10, 5, 8, 6, 15, -4, -15, -10, 14, -1, 0, 4, -8, -3, 7, 13}
, {7, -10, 11, 13, 11, 4, 12, 11, -14, 3, 3, 17, -11, -2, -3, 3, -6, -2, 7, 10, 10, -8, -17, -12, 8, -6, 3, -14, -3, 11, -5, 0, 14, -6, -15, -7, 5, 17, 13, 6, -18, 6, -3, 4, 5, -3, 7, 15, 7, -4, 6, 7, -2, -13, 5, 13, 9, 11, 4, -9, 11, 12, 15, -7}
, {-12, 7, 2, 14, 9, 16, 10, 8, -1, -2, 0, -13, -3, 8, 10, -6, 9, -11, -6, -13, 6, 5, -10, 12, 4, 6, 0, -6, -18, -2, -17, -12, -14, 2, 8, -7, 9, -3, 7, 0, -16, -2, 3, 7, -5, 7, 17, 14, -3, -11, -8, 14, -14, -4, 12, 2, -11, 11, 9, -5, 7, -8, 3, 12}
}
, {{-12, -3, 12, -9, -8, 4, 10, 4, 9, -4, -5, -3, 5, -13, 4, -14, 7, -13, 1, -11, 14, 7, -5, 4, -1, 11, -11, -3, -11, -11, 12, -5, -9, 12, -10, 17, 4, -9, 8, 7, 3, 13, 6, 9, 6, 6, 1, -5, 14, 7, -8, -7, -8, 6, -5, -7, -10, -9, 16, 5, -11, 11, -1, 9}
, {-12, -3, 10, -5, 11, -3, 9, -17, 5, 10, -7, -2, -9, 18, 16, -3, 7, -3, 14, 0, -14, 8, 9, 8, 5, -10, -2, 13, 8, 2, 15, -13, -6, 16, 7, 3, -5, -11, 8, 12, 2, -12, -6, 6, -3, 5, 6, 7, 8, -1, -1, 16, 14, -10, 13, -2, -4, 3, 5, -9, 9, -9, 7, 15}
, {-2, 14, 3, -8, -14, 8, -4, -8, 9, 6, -7, -6, -15, 15, 8, -5, 8, 10, 14, -16, 7, -6, -8, -10, 7, 12, -9, -6, 7, 11, 5, -10, 11, 1, -6, -6, -2, 3, -7, 0, -11, 6, 9, 0, 9, 11, -11, 4, -2, -9, 1, -3, -2, 14, -2, -11, -5, -9, 4, -12, -7, -4, 5, -13}
}
, {{-10, 14, -13, 1, -6, 9, 7, -1, 1, -15, 0, -5, 7, -5, 12, 8, -8, 14, -16, -16, -14, 9, 7, -3, -13, 14, -6, -3, -2, -9, -12, -9, 16, -13, -12, -4, 11, -6, 12, 6, -10, 2, -11, 9, 5, -3, -9, -16, 2, -11, -3, -7, 8, -5, -14, -11, -15, 11, -3, -2, 13, -13, -7, -4}
, {9, 12, -14, 10, -13, -4, 15, 12, -2, 3, 10, -1, 16, 3, -5, -6, 4, -11, -9, 3, -8, -1, -2, -6, 4, -3, 10, 3, -3, 9, 16, -3, -1, 0, -16, 3, 9, -7, -11, -14, 16, 12, 6, -12, 14, 15, 2, -2, 19, -4, -10, -10, 10, -9, -9, 12, 4, -4, 7, -11, -2, 10, 3, -11}
, {3, 2, -12, 15, 13, 7, 8, 8, -13, -14, -2, -10, -5, -6, 12, -15, -11, -5, -13, 2, 4, 1, -6, -3, -15, -6, 10, -8, -3, 12, -11, -5, -1, -1, 3, 4, 13, -2, -12, -13, 15, 0, -7, -15, 10, -8, 9, -13, 3, 5, 7, -6, 10, 8, -15, -14, -7, -1, 11, 6, 6, 2, -9, -17}
}
, {{-1, -9, 5, 3, -15, 10, 1, -6, 4, -2, 8, 2, 15, -2, -13, -15, 3, -4, 13, -2, -9, -8, -7, 12, -14, 17, 3, -17, 6, -7, 0, -14, 1, -10, 4, -14, -1, -8, 13, 16, 12, -2, 6, -12, -10, -11, 0, -15, -11, 6, 12, 3, -17, 14, -11, 3, 4, -6, 12, -2, 2, 0, 12, -10}
, {5, 3, 3, -11, -4, -11, 9, 3, -5, -8, 12, -17, -2, -9, -2, 0, 14, 11, -5, -14, 4, 9, 13, -14, 3, 12, -13, -10, -4, -2, 10, -4, -3, -5, 13, -12, 14, -16, -9, 7, 13, -5, 9, -4, -8, -7, 6, -12, 10, -14, 12, 7, -9, 0, 5, -11, -15, -15, -9, 16, 6, -14, -5, -1}
, {-14, -16, -7, -13, 5, 3, -12, 15, 15, 10, -5, 7, 13, -4, -9, -7, 0, 0, -9, -5, 14, -8, 14, 1, -11, 8, 13, -3, 3, 6, 12, 10, 0, -6, 1, 7, 13, 5, 11, 9, -1, 11, -1, -1, -10, -9, 4, -13, -5, 6, -5, 10, -15, 2, -6, 1, -9, -1, -17, 7, 12, 7, -17, 11}
}
, {{1, -10, -4, 16, -15, 11, 4, -5, -5, -6, 13, 6, -7, 10, -6, 5, 3, -6, -11, 1, 0, -11, -1, 14, -3, 13, 10, -1, 2, -13, -2, -10, 2, 3, -7, -1, 9, 7, 10, 9, 8, 17, -15, 9, 17, -5, 3, -2, 7, 14, -17, -10, -13, 18, -10, -6, 15, -6, -8, -10, 14, -6, -6, -3}
, {10, 14, 3, 8, -4, -6, 13, 12, -1, -4, 12, -13, 14, -14, -1, -4, 4, -17, -3, -8, -11, -16, -2, 4, 9, 14, -13, 4, 2, 14, 2, -6, 9, -8, 10, 16, 14, 9, 13, 1, -4, 4, -17, 9, -7, 4, 15, 10, 13, -4, -2, 16, -8, -9, -15, 13, -11, 17, 1, -10, 18, 13, -6, -4}
, {2, -7, -11, -1, -4, 13, 14, -13, 3, -16, -14, 0, -2, -6, 7, -6, -6, 10, 17, -15, 16, 7, 11, 14, -1, 8, -11, 15, -18, 10, 1, -8, -16, -9, 6, 17, -13, 10, -6, -10, -4, -11, -15, -12, -14, -2, 5, -4, 17, -11, -2, -2, 11, 0, -6, -7, 4, 7, 11, -14, -8, 3, -2, 2}
}
, {{11, 6, -8, 7, -12, -6, -2, 1, 12, 7, 4, 12, 0, -7, 2, 18, 10, 9, -15, -7, 5, 10, 2, -9, -4, 5, -10, 10, -3, -7, 0, 6, 4, -8, 7, -4, 17, -4, 1, -9, 4, 14, -11, -8, 5, -3, -16, -18, -3, -18, 16, 9, 17, 5, -3, -15, -6, 7, 15, -11, -1, -1, 10, -4}
, {-4, 5, -14, 8, -20, 6, -7, -13, 6, -3, -7, 0, 1, -14, 4, 14, -20, 9, 7, -11, 5, 4, -9, -20, -11, -1, 6, 1, 7, 16, -6, -4, 2, -16, 11, -8, 1, -10, -3, 1, -12, 7, 5, 9, -6, 10, 12, -15, 11, 1, -7, -17, 12, 1, 1, 3, 16, 14, 2, -2, 5, -12, -8, -5}
, {-10, 11, 2, -5, -3, -8, 12, -6, -4, -2, 3, -15, 3, -4, 2, 8, 7, 6, -6, 9, -9, -4, 7, -14, -1, 16, 0, -18, 20, 6, -6, -14, 19, 0, -6, 8, -5, 9, -19, 6, 1, -11, -9, -9, 11, -18, -11, -16, 3, 11, 12, -9, 15, -15, 0, -2, 9, 0, -14, -6, -17, -8, -21, -7}
}
, {{-15, 18, -9, -4, 2, 10, -8, 20, 12, 9, 4, 2, 1, -15, 6, 13, 8, -8, 8, -15, -11, -7, 6, -7, -4, 9, -18, -14, -8, -5, -10, -2, -3, -1, -6, 9, 0, 3, 0, 12, -14, 17, -12, -1, 1, -15, 5, -1, -14, 1, 12, 2, 7, -5, -13, -18, 16, 5, -3, 8, 18, 2, -15, 16}
, {-6, 12, -2, 14, -12, 13, -6, 18, -11, 3, -9, 2, 13, 4, -3, 11, -2, -10, -9, -12, -6, 7, -2, 5, -9, 6, -10, -5, 0, 3, 13, -4, -4, -8, 1, 3, -9, 13, -2, 6, 6, 14, -8, 4, 11, -17, -3, -11, 1, -2, 0, -14, 1, 14, 4, 0, -11, 5, 0, 3, 12, 16, -10, 7}
, {6, 4, -8, 13, -16, 4, -12, -7, 7, -1, -8, 13, 3, 6, -8, -11, -1, 8, -3, 2, -11, 0, -6, 2, 8, -12, -14, 10, -11, -13, 13, 13, -10, 4, -12, -13, -9, -12, -3, -14, 10, -5, -9, 5, -5, 1, 13, -1, -11, -14, -7, -3, -6, 10, 4, -9, -12, -11, -6, 6, 7, -6, -10, 14}
}
, {{11, -15, -5, 9, -10, 4, -13, -7, 14, 13, -1, 11, 11, 1, -11, 7, -4, -13, -4, 1, -14, -4, -9, -9, -9, -9, -8, -12, 2, -6, 13, 6, 4, 15, -9, -1, 16, 17, -12, -8, 7, 10, 11, 8, 2, 2, 0, -12, 5, -3, -12, -9, 12, 0, 16, 15, 13, 18, -13, -4, 14, 1, 17, 9}
, {7, 8, 13, -14, 5, -4, -13, -3, -10, -9, -3, 8, 3, -14, -5, 6, 4, 1, 12, -3, 6, 3, 1, 5, -3, 14, -3, 4, 14, -12, 10, 9, -1, 9, 15, 4, -1, 12, -3, -2, 8, -12, 3, -8, 3, 14, 5, -4, -12, 0, -14, -7, -2, 12, -8, -12, -12, 18, -8, 0, -12, 10, -6, 7}
, {5, 7, 6, -12, 6, 3, -3, -14, 5, -2, 0, 8, -15, -14, 9, -14, -1, -8, 7, -11, 3, -14, 12, 4, -12, -12, 6, 13, -8, 7, -1, 6, 10, 16, 3, 17, -2, 2, -14, 8, -5, 3, -2, -10, 12, -13, -14, -3, 10, 13, 7, -13, 5, -5, 14, -11, -7, -3, -14, -3, -8, 16, 6, 14}
}
, {{12, -3, 14, 5, -6, 1, 0, -4, 16, 5, -16, 14, 18, -2, -4, -4, 11, -14, -12, -19, -12, -10, 5, 10, -14, 1, 15, -8, -10, -7, 5, 18, -18, -1, -8, -10, 13, 3, 10, 1, -7, 9, -13, -12, -16, 12, -18, -6, 2, -14, 14, -16, -6, -11, 8, -6, -13, -9, 4, 4, -10, -2, 10, 5}
, {7, -5, -14, -9, -10, 6, -9, 1, -8, 5, 15, 0, -1, 4, -13, 0, -5, -3, 10, 1, -13, 13, -1, -2, 7, -12, -6, 2, 11, 11, 0, 13, 8, 10, 2, -7, 2, 16, 9, 4, 14, 10, -16, 5, 1, 12, 12, 10, -2, 6, 9, 14, 15, 6, -7, 14, -3, 11, 11, 1, -1, 13, -6, 12}
, {6, 12, 7, -3, 3, -3, -12, -3, -4, -11, 12, 9, 9, 1, 1, 4, -14, -3, -6, -14, 3, 8, -14, 10, 9, -10, -11, -16, 13, 2, 17, -16, 9, 11, -4, -6, 14, 14, -9, -2, -4, 11, 8, 15, 9, -7, -7, 7, 16, 5, -5, 10, 10, 2, 10, 12, -2, -6, -11, -15, -14, -8, 12, 6}
}
, {{18, 7, 5, -15, 14, 2, -12, -1, 18, 8, 15, -10, -1, 5, -7, -12, -4, -3, 15, -9, 2, 6, 0, -7, 15, -13, 16, -16, 7, -13, 5, 0, -4, 17, 8, -5, 4, 1, -6, -8, 2, 11, -6, -8, 2, -6, -16, 13, 7, -13, -5, 3, 10, 8, 12, 0, -2, 3, -7, -13, -12, -10, 16, -7}
, {11, -4, -9, 7, -18, 11, 18, 5, 11, 11, 16, -10, 8, -6, 6, -15, 11, 1, -1, -7, 2, -8, -2, -14, -11, -12, 13, -4, -10, -6, 16, 2, -14, 2, 7, -14, -7, 9, 10, 6, -6, 5, -12, 0, 12, -11, -9, 4, 4, 12, 1, -2, 3, -13, -3, -9, 9, -4, -8, -10, 3, -11, 5, -5}
, {2, -12, 13, -14, -9, 16, -2, -7, -1, 14, 15, -5, -14, -14, 5, -14, -11, -12, -6, -13, -4, -9, 14, -4, -10, 4, 2, -17, -5, 1, -12, 9, -4, -4, -4, 10, -5, -11, -4, 8, -11, -11, 6, 3, -3, -8, -2, 0, -6, 14, 1, -12, -16, 6, 1, 14, 9, -3, -10, 1, 10, 7, -9, 7}
}
, {{12, 3, -2, -14, -7, 16, -2, 11, -2, 4, 14, 14, -1, 2, -14, -10, 9, -2, -15, -12, 16, 1, 8, 19, -12, -2, -7, 12, 5, -1, -5, 7, -17, -10, -1, -6, -6, 16, 0, -3, -10, 11, 0, -13, -3, -12, 1, 14, -16, -16, -10, 3, 8, 8, 6, 2, 3, 10, 13, 9, -5, -6, -12, -13}
, {-18, 12, 7, -3, -4, -1, -6, -16, 11, -5, -7, -8, 12, 5, 7, 2, -10, 14, -12, -14, 9, -5, -2, 17, 14, -1, -19, 13, -9, -15, 13, -5, -3, -12, -11, -15, 6, 9, 9, 9, 2, -11, 11, -10, 8, -8, -7, -5, 6, -6, 5, -10, 7, 1, -3, 13, -17, -12, -4, 14, -11, -8, 10, 7}
, {-11, 12, 14, 5, 13, 10, -1, 9, -4, 5, 2, 16, -6, 0, -18, 14, -6, 1, -7, 10, -4, 10, -8, -13, -13, 2, 0, 3, 4, 12, 11, 4, 10, -5, -3, 5, 17, 15, 5, -14, -5, -4, 18, 13, -6, -11, 7, -4, -15, 14, -15, -4, 6, 9, 11, -3, -17, -4, -12, 8, 0, -9, -6, 5}
}
, {{0, -16, 1, -16, 8, -2, 15, 10, -8, 4, -1, 10, 11, -4, 12, -11, -2, 13, -16, 0, -7, -19, 3, 13, 9, 12, -5, -6, -17, 3, 15, -3, -14, 15, 2, 11, 4, -6, -8, -3, -4, 10, -6, 0, -15, -6, 10, 0, -10, -2, -11, -8, 13, -7, 0, 15, -14, -11, -8, 13, -1, 1, -7, -10}
, {-9, -13, -3, -1, 3, 8, -7, -11, 8, 7, -1, -4, 15, -5, 6, -14, 12, 13, 9, -6, -13, -11, 4, 8, 10, -6, 0, -16, -11, 8, 6, -17, -6, 1, 11, 14, -1, -12, -4, -12, -4, 9, -12, 9, -18, -13, -4, -11, -1, -5, 1, -12, 17, -6, 3, 11, 12, -9, 14, -3, 10, 5, 14, 1}
, {-7, -4, -10, -9, 13, 9, -13, 4, -4, 1, 13, -7, 15, 10, -8, -11, 12, 5, 7, 0, 7, -12, -15, 10, 6, 5, -6, 8, -9, -16, 14, -13, -1, -9, -16, 9, 2, 12, -13, 15, 5, 6, -7, 2, -7, -5, -9, 14, 8, 5, -6, -2, -10, 0, 15, 2, -11, 11, 4, 7, 14, -17, 14, 7}
}
, {{8, -12, 8, 9, -9, -6, 10, -12, -13, 4, -5, 12, 8, -5, 4, 1, 7, -9, -3, 12, -15, 1, -12, 2, -5, 7, -20, 13, -21, 9, 8, -16, 11, -16, 5, 10, 8, 3, -10, 3, -8, -7, -2, 16, -8, -4, -3, 4, -10, 4, 7, 3, 5, -8, 16, 4, -10, 15, -6, 3, 11, 14, 12, -20}
, {12, -12, 2, -10, 16, -9, 9, 2, 4, 3, 8, 0, -5, -9, -14, -18, -10, -14, 13, 8, 1, 8, 5, 4, -16, 0, -3, 10, -7, -13, -2, -8, -11, 13, -13, -1, 0, -14, 2, -12, 5, -12, 8, 3, -5, -12, 18, 7, 0, -2, -16, -14, -15, -2, -13, 3, -3, -12, 11, -10, -6, -1, -5, -1}
, {-17, 0, 8, -10, 7, -11, 8, -15, 1, 10, -9, -2, 1, 1, 6, -4, -10, -4, 2, -2, -7, -12, 7, -6, 4, 13, -14, -1, -2, -1, -4, -11, -14, 3, -11, -11, -11, -15, -2, -11, -11, 0, 2, 4, 9, 10, 9, -5, 14, -11, 13, 2, 1, 7, -6, 14, 10, 0, -13, -3, 16, 10, 10, 0}
}
, {{-4, 10, 13, -13, 7, 14, -7, 8, -13, -7, 0, 7, -2, 3, -5, -12, 2, 7, -14, -3, 6, 5, 9, 11, -12, 11, -15, -4, 0, -8, 13, -17, 1, 5, 12, 2, -15, 10, -6, 5, 12, -1, 18, -12, -20, -11, 11, 12, 8, 3, -8, 9, 2, -2, -1, -7, -20, 7, -5, -14, -10, -14, 6, -10}
, {7, -4, -3, 5, -11, -8, 15, -14, 12, 7, -5, 17, -6, -2, 13, -9, -15, 14, -2, -5, 2, 12, 11, -5, 3, -16, -15, -1, -9, 9, -8, 8, 4, 7, -3, -7, -8, -8, 16, -2, 9, 12, -2, -9, -9, 4, -13, -2, -1, 5, 7, 10, -1, -17, 2, 17, -7, -2, -5, -16, 3, -6, -8, -14}
, {7, -12, 7, -16, -1, -13, -10, 3, 5, 7, 2, 4, -12, -13, 14, 9, 2, 2, 2, -13, 7, -11, 3, 18, -3, 14, 1, -14, 5, 10, 7, -17, 4, -17, 8, 1, -9, 1, 11, -9, 14, -5, 5, -1, -1, 0, -14, -5, 5, -5, 8, -14, -9, -10, 0, 14, -17, 7, -5, -5, 15, 13, -1, 13}
}
, {{-13, -12, -1, -3, 14, -2, -15, 7, -11, 5, -17, -10, 1, 15, 13, -12, -16, -4, -16, 3, -5, -2, -5, 1, 9, -6, 13, -9, -10, 0, -15, -2, 15, 9, 1, 2, 3, 2, 15, -15, 8, 2, -9, 13, -5, 12, -3, 8, -5, 7, 2, 0, -13, -15, 2, -3, 6, -3, -16, 0, 15, -8, 12, -10}
, {10, 7, 10, 12, 19, -4, -15, -12, 12, -5, 11, 12, -16, -1, 0, -15, -14, -9, -1, 8, 0, -17, 8, 9, -3, 3, 6, 6, -9, -13, -13, -5, -2, 4, -6, -7, -6, 15, -8, 15, -5, -12, -6, -5, -10, 7, 14, 3, 6, 3, -11, -12, 8, 13, 12, -7, 13, 0, -8, 13, -6, 10, 8, 9}
, {-12, -10, 3, -1, 18, 4, -16, -5, -7, -7, -10, -6, -4, 11, -16, -16, 9, -7, 1, 8, 12, -20, 3, 3, 9, 9, 12, -5, -9, 11, 3, 6, -4, -3, 6, 1, 3, -4, 12, -9, -10, -16, -10, -10, 3, 10, 0, -15, 4, 11, -10, -6, -9, 1, -16, 12, 8, 2, 0, 11, 4, 11, -14, -11}
}
, {{-9, 2, -8, -15, -10, -9, 5, -7, 10, 0, -13, 8, -7, -17, -2, 7, 6, 15, 3, -4, 14, 2, -2, 4, -14, -9, -18, 10, 18, -12, 21, -9, 4, -5, 3, -4, 2, 10, -3, -1, -1, -13, 1, -8, 7, -6, 14, 5, 10, 8, -7, 13, -10, 3, 2, -7, -5, -1, 4, 0, -12, 15, -6, -16}
, {2, -2, 8, 9, -17, -5, 8, -1, -10, 4, -11, -7, 7, -7, -10, -12, 6, -7, -18, 15, -9, -15, -15, 12, -14, 0, 13, 6, 19, 15, 18, 11, -8, -13, -21, 16, -13, -1, -17, -10, 4, -6, 3, -11, 3, 2, 6, -3, -10, -4, 0, -12, 12, 4, -19, -7, 2, -14, 9, -7, 13, -15, -4, 8}
, {9, -7, -12, 9, 7, -4, -3, 13, 2, -9, 1, 5, 12, -8, -13, 2, 0, -11, 9, -13, 4, 12, 9, 2, -3, 2, -1, 7, 16, 1, 2, -17, 0, -11, -18, 3, -9, -4, 3, 12, 0, -2, -13, 6, -8, -17, 0, 7, 19, 6, 5, -8, -12, -8, -16, 3, 16, -1, 3, -14, -4, 4, 4, -3}
}
, {{1, -7, -1, 14, -12, -13, -9, 6, -4, 3, -1, 10, 16, -14, 8, 3, 14, -2, -7, -11, 7, 13, -7, 2, 0, 2, -4, 2, -6, -6, 7, -6, -17, 10, -5, 1, 10, -16, 4, 2, 14, -16, 0, -6, -9, -16, 2, -10, -17, 6, -5, -1, 4, -7, -15, -8, 10, -7, 0, -9, -12, -1, 7, 0}
, {1, -10, -1, 8, 12, 7, 1, 5, 6, 7, -4, -11, 10, 11, 2, 1, -18, -4, 15, -6, -3, 5, -13, -3, 12, -13, -9, 15, 6, -14, -12, 0, 2, -16, 6, 2, -12, -3, 18, 12, 7, 0, -18, 2, 3, 10, -2, -10, -15, -5, -18, 12, -19, -5, -11, 8, 7, -6, -11, -13, -5, -11, 6, 13}
, {6, 12, 9, -10, 12, 2, 1, -7, -16, 6, 3, 1, 1, 15, -10, 12, 4, -6, 12, -15, -11, 0, 7, 7, -5, -1, 6, 8, -15, 4, 8, 15, 4, 9, 15, -16, -9, 5, 18, -8, 9, 8, 9, 16, -5, 7, 14, -3, -2, 12, 12, 3, 2, -13, 8, 4, 0, -3, 8, -15, -1, 1, 9, -12}
}
, {{8, 3, -10, 0, 1, 9, 5, -6, 7, 4, 14, -5, 9, 4, 10, 3, -17, -13, 12, 17, 7, 5, 1, -14, 7, -10, -4, 12, -15, 8, -10, 1, -11, 12, 1, -10, -8, 16, 17, -8, -10, 6, 2, 3, 6, 12, -4, 2, -2, -14, -1, 3, -14, -4, -3, 6, 0, -5, -7, -10, -14, -1, 0, -4}
, {-5, 1, -4, -6, -11, -16, -16, -1, 13, 14, 6, -7, -12, -3, -5, 15, -14, -9, 10, -9, -12, -9, -13, 14, 19, -9, -4, -1, -10, 7, 1, 6, 17, -12, -14, -14, -3, 15, 3, -11, -6, 12, -4, 10, -5, 15, -7, -2, -5, 6, 10, 11, -3, -5, -5, -11, 20, -8, -10, -3, -12, -15, -11, -9}
, {8, 3, 12, 5, -8, 4, 5, -10, -2, -3, -6, 13, 1, 3, 12, -11, -14, -8, 2, 4, -19, 12, 3, -10, 21, 15, 12, -13, -9, 5, -1, 13, 7, 9, 3, -7, -1, -1, 2, -2, 6, 10, 4, -10, 14, 3, -4, -12, 3, 13, 7, -7, 13, -3, 0, -13, 6, -12, -11, -4, 4, -11, -6, -3}
}
, {{-7, -13, -17, 2, 8, -14, -3, -9, 9, -13, -10, 4, 2, 7, 4, -5, 9, -8, -2, -6, 2, 6, 5, -7, 13, -4, 15, -3, 3, -7, -9, -14, 10, 8, -8, -3, 18, -5, -9, -9, -5, -3, -7, -16, -11, -13, -9, 10, -16, 14, 4, -9, -16, 15, 12, -16, -13, -8, 14, -7, 15, -5, 5, -17}
, {-7, -12, -8, -1, 11, -11, 11, 5, 1, -17, 13, -7, -1, 9, 1, -6, 8, -1, 2, 9, 1, -3, 4, 17, 12, 2, 2, -1, -10, -11, -6, -16, 6, 3, -5, -1, -15, -8, 1, 6, 9, 5, -7, 2, -5, 15, -16, 2, 3, -14, -9, -13, -15, -8, -4, 13, -9, -14, 11, 6, -3, 1, -15, -1}
, {2, 9, -10, -1, 16, -4, -8, -17, -9, 2, -2, -17, -15, -9, 10, -12, -1, 7, -1, 12, 12, -2, -3, 5, -16, -15, -3, -14, -5, 1, 8, 6, -5, -11, 2, -5, 4, 8, 4, -12, 6, 3, -8, -5, -18, 0, 0, 8, -12, -1, 15, -1, -13, 13, 11, -15, -15, -11, -7, 6, -9, 2, 10, 2}
}
, {{4, 2, 16, -13, -14, -13, 0, 11, -8, 7, -12, 10, 8, -6, 6, -11, 16, 0, -8, 2, -6, 16, 7, -6, -17, -4, -2, 3, -9, -6, 9, 2, -15, 4, -2, -16, -3, -11, 13, -14, 15, -13, -3, 1, -13, -3, -13, 12, -10, 3, 8, -1, 7, 1, 0, -6, -17, -16, 9, -11, 12, 13, -12, -2}
, {-8, -5, 3, -11, -4, -15, 5, -6, -5, 13, 5, -6, -2, 14, -3, 0, 6, -16, -10, -14, 2, -6, 15, 0, -12, 10, 4, 13, 3, 8, 3, -17, 4, -8, 9, -7, -10, 16, 9, -8, -8, 12, 0, -13, 13, 13, 11, 9, -1, -10, -1, -11, -2, 4, 7, -5, 2, -12, 10, 1, -6, -11, 0, -10}
, {12, -5, 14, 13, 9, -5, -11, -17, -17, -16, 13, -10, 4, 7, -3, -6, -8, 9, -13, 3, -15, 2, 7, -6, -7, -3, 2, 6, 8, 11, -5, 7, 0, -6, -6, -1, 6, -17, -2, 2, -4, 14, 0, 10, -12, -12, 11, -13, -12, 12, -6, -5, 7, 12, 10, -12, -3, 8, 6, -6, -10, 11, -10, 11}
}
, {{7, 10, 3, -2, -2, -15, 6, 15, -13, 9, 1, 13, 5, 10, 2, 12, -8, -5, 6, -11, 0, -15, -1, -5, -8, -11, -5, -5, 17, -14, 0, 9, 1, 8, 5, -7, -4, -2, 7, -16, -16, -5, 5, -1, -16, -1, -1, 4, 13, 3, 9, -6, 1, -15, -9, -9, 18, 10, -15, -2, 6, 8, -16, -9}
, {14, 10, -4, 7, 7, 9, 8, 11, -15, -16, 14, 9, -9, 12, -6, 3, 8, -9, 6, -6, 14, -7, -14, -2, -4, 7, -14, 17, -8, 8, 10, -16, 1, 15, 2, -14, 11, 3, 13, 4, -19, -13, -9, 8, -11, -11, 0, 11, -2, -11, -5, 12, -15, -15, 0, -12, -4, 4, 10, -3, 15, 13, 3, -6}
, {11, 9, 9, 9, -11, 0, -12, -4, -16, -12, 4, 13, -10, 17, -6, 5, -2, -9, 2, 5, 5, 9, -11, -10, -10, -4, -5, 15, 0, -18, 3, 14, 8, 16, -16, -3, 1, 6, -3, 6, -5, 5, 10, -2, 10, 10, -6, -3, -5, 7, -7, -9, -5, -8, -5, 8, 1, 4, 8, 1, 4, -6, -13, -11}
}
, {{4, -7, 7, 15, 15, 10, 10, 2, 7, 12, -10, 3, 1, -5, 7, 9, -2, 6, 4, 5, -2, -9, 10, -7, -6, 9, -5, 5, -1, 8, -9, -2, 5, 2, 13, -8, 0, -2, 2, -5, -19, -2, -1, -14, 5, -2, -2, 12, 9, -3, 3, 6, -13, -8, 6, 12, -3, 6, -17, -9, 13, -3, 12, 1}
, {-2, 11, 9, 7, 20, -15, -14, 11, -2, -2, 5, 16, 0, -10, 1, -3, -14, 4, -11, -14, -11, 8, 0, 11, -8, 10, -2, 14, 7, -1, -1, -11, 6, -11, -12, 3, 2, 13, 9, 10, -18, 7, -7, 13, 11, -14, 12, -6, -13, 12, -13, -3, -14, -2, 1, 6, -4, 13, 13, -6, 15, 13, 4, -4}
, {7, -16, 15, 1, 16, -12, -8, -18, 3, -13, 17, 15, -5, 5, -11, -10, -7, -1, 15, 2, -6, 0, -9, -4, -15, 2, 2, -7, -11, 6, -6, 7, -15, 11, -13, 2, 5, -12, 3, -12, -9, -15, 6, -7, 7, -13, 1, 6, -16, -15, 10, 4, 10, -3, -6, 0, 5, 15, -9, 15, 0, -11, -4, 5}
}
, {{8, -14, 9, 1, 13, -7, -1, 11, -6, 11, -5, -1, -2, -6, 7, -1, -6, 4, 12, 6, 10, 15, 2, -5, 1, 16, 2, 6, 10, -8, 6, -4, -12, -5, 12, 15, -9, -10, 9, 7, 6, 15, -9, -16, -13, 5, -13, -8, -5, -14, 12, 13, 6, 7, 11, -3, 11, 12, -6, 13, 5, 9, 8, 18}
, {16, 10, -17, 2, 8, -8, -1, -9, 10, 14, 7, 15, 0, 11, 5, 7, -1, -5, -9, -16, -14, -9, 16, -5, 7, 2, 14, -5, 14, 1, -10, -12, 3, -14, -6, -9, 13, -4, 2, -13, 17, -10, 4, 0, -2, -5, 8, 2, -13, -8, 4, 10, -5, 10, -6, 3, 8, 13, 12, 11, 10, 7, -16, -5}
, {13, 4, -16, 12, 1, -9, -4, 15, 12, 7, -9, 4, -10, 6, -10, -9, 6, 8, 0, 5, 12, -5, 8, 4, 15, 4, 6, -13, -8, 13, 13, 13, -6, 12, -8, 1, -12, 8, 5, -5, 10, -1, -15, 10, 13, 11, 6, -16, 4, -8, -14, 13, -6, 11, 4, -1, -15, -2, -3, 6, 5, 11, -3, 8}
}
, {{14, 14, 4, 10, -15, 10, 7, -4, 6, -9, -7, 5, 2, -10, 3, 10, -14, -14, -2, -19, -10, -6, -8, -14, 0, 3, 12, 14, -9, 12, 11, 7, 7, 11, 3, -7, -3, 6, 7, -3, 8, 2, -8, -5, -5, -13, -17, -16, -10, 6, 2, -14, 13, 4, -14, 13, -11, -15, -6, 3, 16, 12, 7, 7}
, {-1, -7, 8, 10, -13, 14, 6, -12, 5, -8, -2, -9, -16, -12, -13, -15, 14, -14, 10, 8, 1, 13, 7, 0, 6, 2, 12, -14, 1, 8, -12, -5, 6, -20, -17, -15, -5, -4, -2, 5, 10, 7, -15, 9, 3, -14, 13, -15, 3, 12, -8, 12, -9, 10, -12, 5, -5, 3, 10, 15, 10, 0, -5, 0}
, {-10, -20, -6, -5, -5, 1, 9, 5, -15, -4, -4, 0, 6, 16, -4, 4, -12, 11, -17, 8, -6, -7, 13, 7, -13, -6, -15, 0, 3, 3, 12, 1, 9, -15, 13, -21, 13, 9, 7, -7, 8, -14, -15, 11, -17, 9, 7, -3, -6, -4, -9, -1, 13, -4, 1, 0, 15, 0, 0, 4, 9, -2, -14, 10}
}
, {{-13, -5, -4, 14, -1, -8, 6, 1, -9, -12, 10, -11, -1, 14, 6, -9, 13, 0, -12, -16, -8, 13, -7, 14, -10, -11, -4, -6, -13, -4, 1, -11, -15, -3, -2, 5, 7, -1, 1, 4, 1, 8, 14, 2, -1, 11, 6, 8, -9, 4, 7, -5, -16, -16, -6, -3, -12, -3, -2, -15, -8, -4, -5, 15}
, {-3, 7, 13, -13, 13, 4, 6, 12, -14, 12, -11, 3, -15, 12, -13, 4, 0, -15, -4, -4, 4, 11, 11, 2, -7, 6, -8, 12, -9, -16, -11, 7, 4, -14, -8, 3, 9, 14, 11, 6, 14, 4, 15, -15, -13, -13, 13, -15, 12, 9, 10, -6, 0, -16, 8, -5, 5, -9, 4, -16, -2, -17, -8, -5}
, {2, -3, 0, 16, 1, -15, -15, 11, -12, -8, -3, -6, 9, 12, 4, 3, 4, -2, -9, -8, -7, 4, 11, 8, 16, 5, -3, 14, -19, 13, -9, -17, 4, 8, 12, 10, -14, -1, 2, -1, -4, -15, -7, -6, 3, 10, 2, -11, 7, 6, 6, -12, 9, 1, -14, 2, -1, 1, -2, 2, 11, -1, 7, 10}
}
, {{7, -7, -19, 6, -3, -15, -10, -17, 4, -18, 4, 12, 8, -15, -5, -10, 3, -10, 3, 5, -7, -3, 8, -13, -1, -16, -15, -9, 0, -9, -2, -4, -13, 0, 0, -8, 9, -11, -2, 4, 0, -16, 3, -10, 2, 12, 6, -12, 3, 3, 14, 8, 7, 5, 3, -15, -15, -2, -4, 12, -11, 10, 6, 13}
, {-13, 0, -1, 0, -7, 2, 1, -15, -5, -20, 7, -8, 9, -14, -14, -5, 13, 1, 6, -5, -5, 4, -10, 19, 4, 4, -17, 10, 10, -9, 13, -16, 13, -16, -7, -14, -11, 16, 1, -14, 6, -4, 17, 16, -3, -7, -8, -6, 3, 14, 16, -17, 5, -15, -7, 16, 10, 10, 5, -3, 9, 9, 13, 4}
, {-3, -13, 0, -4, 8, -13, -12, 6, -15, -17, 4, -2, -5, 12, 11, 4, -2, 23, -8, 13, 10, -4, 1, 13, -14, -16, 14, -3, -9, 11, 0, 5, -15, 6, 1, -5, 2, -13, -7, 0, -4, -14, 0, -6, 3, 7, 8, 8, 8, -1, -1, -14, -6, 10, 5, 11, -10, 7, 6, 5, 0, -17, 14, -19}
}
, {{-2, -9, 0, -7, 20, 4, 12, -11, 10, -18, 14, 3, -9, -9, -4, -1, -5, -8, 13, 4, 4, 13, 16, -6, 9, -12, 2, 1, 7, -7, -14, 6, -10, 7, -11, 0, 10, -2, -4, 5, -3, 12, -14, 16, 10, -11, -6, -3, -2, 7, 7, -16, -10, -1, 12, 13, 5, -17, 11, -5, -4, 9, -8, 4}
, {1, 8, 3, -6, 9, -8, 17, -15, 11, 4, -3, -10, 16, -2, 14, 3, 3, 2, -14, -6, 12, -4, -9, -5, -15, 1, 8, 4, -7, 10, -2, -8, 3, -13, 1, -5, -12, 16, 5, 3, 12, -12, -13, 0, 5, -12, 8, -14, -14, -10, 5, -3, -15, 15, 13, 21, -1, -10, 0, -13, -3, -4, -11, -6}
, {16, -10, -8, 9, -6, -5, 0, 13, 0, -2, -2, 15, -5, 12, 11, -13, -15, -7, 12, 0, 5, -6, -3, -1, 2, -2, -2, -8, -6, 8, 7, 8, -17, -6, -2, -5, 13, -5, -9, 11, 17, 4, -5, 17, -9, -18, 14, -8, 6, -7, 0, 9, -13, -3, 18, 12, -17, -15, 4, 0, 0, 5, 13, 4}
}
, {{-18, -7, -15, -1, -10, -2, 0, -8, 11, 1, 13, -6, -12, 8, -13, 2, -10, -10, -3, 0, -4, 8, -3, 13, -1, 2, 3, 6, -3, -3, 1, 10, -3, -10, -1, -9, 1, 0, 0, 12, 5, -3, -6, -9, -17, 8, -5, 3, -15, -8, -11, 11, 9, -11, 6, 3, -15, 5, 4, -11, 2, 2, 11, 5}
, {-16, 12, -7, 2, 8, 18, -3, 3, -6, -14, -2, 2, -7, -1, -13, 11, -10, 0, -12, -5, 10, 11, 1, 2, -17, -14, 10, 11, 4, -9, -3, -14, -7, -16, 12, 16, -1, 9, 10, -13, 5, -17, -8, 4, -15, 2, 6, -12, 12, -14, -5, 5, 8, -11, 3, 12, -17, 6, 10, -10, 0, 1, -16, 10}
, {-16, 6, -12, 3, 6, 3, 16, 5, -12, -11, -10, -8, -6, -8, 10, 0, 10, -4, 11, -3, 0, -14, 9, 10, 14, -1, 11, 11, -13, 12, 6, -2, 13, 4, -4, -16, -15, 7, 6, -4, -2, -17, -5, -13, -4, -2, 0, -7, -10, 5, -7, -8, 7, 7, -15, -11, -5, 12, 12, -17, -17, -15, -18, -12}
}
, {{12, -12, 3, 0, 12, -6, 13, 9, 13, 6, 4, 1, -10, -9, 15, 8, -12, -6, 7, 4, 18, -8, -14, 14, 7, 8, 11, 13, 1, -15, -11, -10, 17, 6, -8, 3, 6, 1, -5, 8, -1, -7, 3, 7, 9, 14, 11, 8, 4, -15, 15, 11, -10, 4, 5, 5, -8, 9, -6, -6, 2, 6, 7, 7}
, {-2, -4, 14, -1, 3, 16, -7, 0, 4, -12, -14, 3, 1, 3, 7, -13, -5, -2, -5, -10, 6, 11, 4, 18, 14, -15, -9, -11, 1, 4, -9, 11, -7, 9, -4, 0, 10, 7, 11, -8, -11, -6, 2, 0, -7, 0, 14, -12, 13, -14, -5, -11, -4, -15, 11, 16, -13, -14, 1, -2, -12, -6, -13, -14}
, {-5, -9, 15, 6, 0, -1, 15, 11, -3, -9, 8, 15, -8, 4, -5, -2, 11, -3, -15, -3, -3, 12, -16, 2, 2, -11, -8, -7, -3, 5, 16, -10, -11, 6, -18, -17, 6, 16, 6, 16, 13, -2, -2, 1, -3, -5, 8, -10, -5, 9, -1, 4, 2, 8, -1, 9, 8, -13, 11, -6, -4, -9, 13, -1}
}
, {{-16, 6, 5, 9, -11, -9, -2, -17, -10, 0, -8, 18, -9, -14, 12, -1, 11, -14, -8, -3, 10, 11, 16, -2, -10, 8, -20, 11, 10, -4, -2, 7, -18, -13, -7, -3, -17, 13, -12, 13, 4, 9, 17, -2, 14, -6, 6, 14, -3, -4, -7, -10, -2, -2, 5, -5, -10, -7, -3, 5, 17, -14, -14, 1}
, {-1, 5, 6, 7, 3, 0, -11, -6, 16, 5, 19, 0, 5, -9, -7, 10, -9, -2, -6, 1, -4, 13, -2, 3, -13, -2, 1, -1, 14, 11, 11, 2, 15, -12, 6, -13, 12, -13, 14, -8, -2, -14, 11, 8, -11, 7, 13, -6, -4, 16, 7, -2, 12, 8, 1, -1, 6, 13, 11, 15, -5, 8, 3, 0}
, {1, 8, 12, -10, 4, 7, -8, -1, 8, -5, -1, 13, 12, -12, 16, -17, -5, 18, 10, 0, -2, 10, 11, -8, -2, 5, -1, 8, 1, -1, -6, -8, 0, 13, -2, -5, 7, 16, -1, 7, 3, 13, 8, 6, -5, -10, -1, 6, -11, 12, -12, -5, 8, -5, 4, 17, -6, 6, 0, 0, 12, 4, 11, 3}
}
, {{5, 9, 1, -13, -14, -12, -4, 17, 17, 8, -12, -4, 11, 15, 17, -12, -2, -6, 4, 3, 5, -6, -3, -14, 3, 2, 13, 12, -10, 3, -12, -1, 12, 10, -10, -8, -4, -1, -12, -16, 12, 3, 12, -8, -1, 6, -12, 11, 11, 16, 10, 15, 3, 16, -3, -9, 5, 14, -5, -5, -6, -9, 14, 3}
, {1, -2, -6, 13, 15, -9, 18, 8, 1, -6, -6, -6, 9, 9, 10, 6, 12, 2, 2, -9, 11, -6, -8, -2, 1, -12, -5, 3, -12, 14, 15, -11, -6, 10, 2, -12, 3, 7, 12, 12, 11, 1, 14, 11, 12, -4, -9, -12, 13, 1, -3, 7, 1, -9, 5, -13, 0, -6, -12, 7, -13, -4, 3, -2}
, {-10, 6, -8, 11, 10, 12, -12, 11, 2, 11, -13, 12, 13, 0, -7, -13, -9, 12, 2, 2, -10, -9, 2, 0, -8, 1, 15, -17, 2, 12, -9, -15, 6, 4, 12, 15, 16, -5, 9, -13, 7, -4, -12, 1, 0, -1, 3, 14, -6, 16, -5, -1, -10, 12, 8, -9, 2, 2, -15, 8, 0, -1, 7, -13}
}
, {{10, -2, 5, 15, 12, -2, -7, -8, -13, 1, -2, 3, 12, -12, 10, -5, -5, -4, -13, -1, 14, 7, -4, -7, 11, 6, 5, 5, -13, 2, -11, -7, 1, -9, -6, -12, 5, 12, -1, -13, 5, 12, 7, -13, -1, -16, -16, -2, 7, -9, -8, 10, 8, 0, -8, -10, -2, 4, -14, 13, 7, 2, -2, -12}
, {1, 2, 5, -4, -5, 13, -8, -2, -7, -17, 15, 14, -4, 8, 4, 0, -1, -18, -8, 19, -11, -13, 12, 3, 7, -9, -2, -13, 3, 7, -13, -2, 13, 10, -16, 13, -18, -10, 3, 0, -12, -2, 2, 12, -10, -13, -11, 5, 6, -11, 11, 16, 9, 5, -11, 15, -3, 13, -15, -15, -8, -11, 5, -12}
, {-3, -12, 5, 7, 7, 11, -10, 6, 12, 0, 5, 0, 5, -13, 15, -2, -10, -3, 11, -4, 1, -1, -16, 12, 3, -14, 1, 7, -1, 1, -8, -6, 13, 18, 1, -13, -10, 7, -13, -8, 7, -13, 0, 14, -12, -6, -15, -8, -1, 8, 15, 6, 11, -10, -17, -1, -4, 14, 14, 13, 16, 2, 1, 9}
}
, {{-11, -14, -7, 14, -8, -16, 0, 0, 13, 7, -13, 11, -12, -2, 10, -3, 13, 3, 0, -13, 10, -8, -3, -10, -6, -8, -7, 14, -6, -8, -12, 12, -3, 15, 0, -4, 0, -3, 11, 14, 1, -11, -12, -2, 8, 12, -11, 6, 11, 13, -9, 2, 0, 1, 10, 5, 8, -7, -4, 2, 0, 0, 1, 9}
, {-7, -7, 9, 0, 11, -9, 5, 2, 6, 8, -14, -11, 10, 10, 10, -7, 14, 6, -8, -6, -17, 2, -11, 5, -12, -11, 3, 6, 11, -1, 5, 0, -5, 11, 14, -13, -2, -2, 10, 13, -13, 6, -5, -6, 11, -3, 9, 15, 12, -7, 0, -1, -6, -11, 3, -4, 11, 4, 3, 19, 12, 4, 16, 1}
, {-9, 0, -16, -15, -5, -15, 15, -11, 9, -7, -2, 6, 11, 13, 16, 6, 12, 10, 6, 4, 1, -14, -1, -4, 6, 7, -2, 8, -2, -4, 7, 5, -3, -10, 9, -6, 18, -12, -8, 13, -6, 4, 6, 9, 11, 2, 0, 4, 7, 8, 13, -14, -7, 11, 10, -5, 4, -9, -8, 2, -9, 2, -8, 2}
}
, {{11, -11, 0, 9, -12, 3, -8, 6, 3, -4, 14, 5, -4, -7, -10, 6, -1, 5, -8, -5, 1, -3, 13, 0, -1, -8, 16, -18, -3, -12, 0, -2, -12, -8, 2, 12, 3, 13, -10, -2, -2, -17, -10, 12, 11, -15, -11, 4, 3, -2, -6, 13, 13, -11, 5, 17, -12, 4, -5, -11, 4, -5, -15, -10}
, {-10, 6, -17, 0, -13, 2, -11, 6, 12, 7, -5, -11, 0, -2, -17, -1, 0, 1, -3, -12, -6, 6, 10, 6, -9, -8, 13, -15, 5, -6, 12, 1, -13, -3, 11, 15, 15, -1, -2, -9, 16, 13, 9, -6, -9, -14, 4, 11, -3, 14, 3, -9, -10, 11, -5, -7, 0, -2, -10, -3, 12, 7, -7, 13}
, {-7, -3, -13, 4, -4, 12, -1, -1, 12, 0, -11, -8, 9, 11, -8, 3, -11, 12, -17, 8, -19, -6, 15, -2, 10, -9, 7, -2, -9, -8, -2, 8, 12, 10, 9, -4, 6, -5, -10, -2, -2, 7, -8, -6, 5, -11, 1, -7, 11, -14, -19, -3, 4, -8, -12, 9, -1, -1, -13, 2, -1, 2, 0, -14}
}
, {{-13, -2, -16, 9, -6, 7, -4, 5, 1, -2, -7, 4, 14, -10, 14, 0, -7, -1, 9, 6, 6, 11, -8, 9, -1, 0, -11, 3, 2, 2, 17, 15, 15, 2, -9, -13, -2, -2, 3, 6, -15, 3, 5, -16, 0, 10, 16, 9, 14, 12, -8, 0, -2, -9, -8, 11, 19, 2, 4, -5, 3, -14, -11, 18}
, {-2, -5, 2, 15, 23, 13, 1, 11, 6, -10, -1, 4, 13, 4, 3, 4, -5, 4, 6, -14, -7, -8, 3, -2, -10, -3, 8, -14, 19, 10, -1, 4, 5, -5, 0, 1, 0, -7, 4, -10, 8, 16, 3, -11, 7, 17, 21, -6, 4, 9, 14, 11, -10, 14, -16, -16, 16, 6, -12, 6, -12, 13, 8, 11}
, {15, -3, -11, -15, 5, -17, -1, 8, 6, 13, -14, -1, 11, -8, 16, -4, -2, 6, 4, -9, 7, 7, -11, -15, 3, 17, 11, -1, -7, -11, 5, 2, 10, 5, -14, -5, 4, -9, -3, 1, 4, 8, 4, -17, -1, -5, -11, 5, 15, -5, -5, -11, -2, 17, -3, 2, -9, 10, 12, 7, -12, -10, 14, 7}
}
, {{-4, 3, -5, -12, 11, 5, -12, 1, 11, -14, -14, 1, -8, 14, -5, 7, 10, -13, 8, -1, 6, 6, 11, 6, 0, 7, -7, 4, 5, 12, -10, -15, 11, 9, -14, -19, -13, -1, -18, -9, 6, 10, 13, -2, 0, 4, -14, 17, -1, 5, 6, -1, 12, -1, -15, -4, 14, 14, 7, 1, -7, 6, -1, 3}
, {11, -6, 7, -8, 13, -11, 14, 1, -10, -9, 2, 0, -7, 7, -8, -11, 6, 7, -10, 5, 15, -14, 3, -7, 0, 5, 12, 8, 12, -11, 9, 9, 14, 3, 10, -18, -12, 7, -9, -8, 11, 9, -2, -12, 9, -6, -6, 13, -11, -13, -9, 6, -6, 9, -1, -1, -4, 4, 0, 14, -14, -13, -3, 11}
, {-10, 8, -10, -15, -3, 7, 13, 3, 5, -13, 6, 8, -13, 5, -5, 12, -3, -13, 13, -3, -16, 9, 1, 7, -2, 7, 10, -10, 10, -14, -5, -3, -9, 17, 13, -13, -9, -6, -3, 13, 1, 14, -7, 7, -18, 9, -10, -8, -10, 16, -5, -5, -15, 2, -16, 4, -9, 15, 9, 0, -8, 12, 9, -15}
}
, {{-9, -1, -8, -3, -3, -10, 10, -8, -17, 5, -7, -14, 9, -12, 3, -14, -13, 0, -16, -16, -14, -9, -2, 2, -5, -9, -5, -16, -16, 8, -10, -11, 5, 7, -2, 0, 7, -8, 13, 3, 2, 3, -7, 12, -5, 12, 0, -12, 15, -7, 10, 4, -7, -13, -16, -13, -9, -1, 12, 3, -15, -11, -15, 6}
, {-1, -6, -9, -4, 3, -13, -4, -3, 3, 11, 7, 3, -5, 5, 7, -5, -8, -4, 1, 8, 15, -15, 8, 4, 2, -17, 9, -1, 9, 5, -2, -11, 1, -15, -16, 5, 2, -12, 6, -13, -14, 7, -13, 0, -11, -6, 12, 10, 14, -3, -10, 13, -3, 3, 0, -19, 3, 15, 8, 6, -1, -14, 9, 11}
, {9, -10, -7, -3, 0, 9, -5, -9, -4, 8, 2, -16, 10, 4, -17, 8, 1, 3, -5, 9, 15, -5, -10, -1, -2, 2, -13, 7, 4, 6, -13, 9, 4, -4, 8, -2, -15, 8, 1, -1, 13, 2, 11, -14, 4, 2, 7, 4, 0, 5, -2, -9, 15, 0, -6, 4, 7, 11, 4, 2, 15, 3, 9, -13}
}
, {{10, -10, -10, -9, 12, -12, 8, 11, 10, 15, 3, -1, 0, 10, -12, 2, 13, 12, -9, 1, 5, 6, 15, 12, -7, -10, -1, -9, -1, -6, 3, -6, -11, -4, -6, 14, -5, 5, -9, 17, -8, -7, -4, 10, -13, 2, -3, -16, -14, -6, -1, -11, 4, -11, 2, 0, 17, 15, 6, 8, 11, 5, 7, 16}
, {-9, 13, -1, -2, 5, 15, 4, -11, 14, 10, 7, 5, 11, -1, 3, 12, 10, -15, 1, 1, 1, -14, -10, -16, -8, 7, 4, -2, 10, -7, 6, -2, 8, -13, -5, -1, 9, -16, 6, -12, 5, 13, 5, -1, 3, 6, 3, 15, 11, -2, 11, -7, -15, -4, 2, 2, 1, 15, -8, 4, -12, -12, -11, 0}
, {13, 2, 0, -8, -8, 14, -15, 4, 10, -4, -3, 4, 14, 0, -5, 2, 7, -2, 1, -4, -17, -8, 3, -15, -2, 10, 2, 2, -3, -11, 13, -4, -1, 13, -5, -15, 2, -12, -16, -2, -16, -6, -3, -4, 16, -2, 0, -4, 9, -4, -13, 13, -15, 0, 8, -9, 5, 9, -3, 12, -5, 13, 12, 5}
}
, {{3, 2, -4, -1, 18, 9, 11, 8, 8, 2, -4, -2, 11, 6, 1, -4, 12, 0, 11, 5, 6, 0, 13, 9, -4, 1, -12, -6, -2, -7, -10, -3, -7, 12, -2, -8, 7, 1, 2, -8, -5, 15, 10, -13, -12, -9, 17, 17, 14, 10, -15, -7, 14, 10, -16, 5, 3, 5, 4, 15, 12, 10, -15, -3}
, {14, 17, 2, 5, 10, 16, -2, 15, 9, 6, 11, -17, -15, -7, -6, -13, 14, -3, 5, 20, -4, -8, 0, -16, 2, -1, -15, -3, 13, 9, -15, 6, -5, -12, 17, 14, 5, -11, 6, -4, -17, -2, 17, -14, 6, -12, 13, -10, -10, -11, 12, -5, -9, -3, -17, -11, 2, 5, 14, 9, 13, -5, -9, -11}
, {3, 6, 0, -10, 10, 8, 16, 1, -10, 13, -4, -15, -8, -6, 5, 18, 7, 3, 8, -7, 4, -10, -12, -9, -9, 1, 7, -6, -11, -8, 6, -1, 6, -8, -1, 5, -13, 12, -16, -8, 3, 8, 16, 8, -8, 6, -13, 15, 14, -3, -16, -12, -7, 16, 10, -14, 11, -9, 4, -3, 7, 11, -16, -12}
}
, {{14, -12, 13, -4, 4, 3, -14, -3, 11, 10, -7, 1, -10, -2, 1, -3, -11, -11, 9, 0, -3, -11, -7, -12, -7, 1, 8, 6, -18, 3, -2, 14, 4, 8, -8, -15, -14, 2, 7, 14, 1, -4, -18, -3, 8, 11, 12, 10, -16, 17, 13, -16, -17, 8, 14, 3, 17, 6, 2, -13, -2, -4, 2, -11}
, {-1, 10, -6, 9, -8, -7, -15, 16, -2, 10, -5, -1, 11, 13, -7, 13, -8, 0, 2, -3, -2, -12, -8, 14, 0, 4, -12, 10, -2, 10, 6, 3, 1, -10, 0, -3, -9, 17, -10, -15, 14, -4, 6, -11, -2, 11, 5, 6, -2, 15, -2, 8, 13, -10, -3, -6, 16, 10, -3, -16, -5, -16, 9, -12}
, {-1, 13, -10, 3, -9, 3, 4, 11, 12, 10, 8, -13, 12, -17, -3, -11, -11, -16, -12, 7, -19, -12, 3, 5, 14, -5, -15, -7, -17, -10, -5, 5, -9, 1, 8, -15, 9, -2, 9, -14, -2, -16, -19, 16, 16, -6, 6, 7, -8, -6, -2, -2, -3, 8, 11, 9, -7, -7, -11, -3, 1, 10, 5, 12}
}
, {{-5, -17, -11, -2, 0, 3, -1, 8, -13, 11, 10, 3, -2, 11, -16, 12, 7, 2, 9, -14, -10, 8, 8, -10, 15, 0, 6, 10, -4, -12, 3, 0, 9, -1, 11, -14, -11, 14, 16, 14, -4, -9, -5, 18, -8, 3, -3, 5, -3, 14, -5, -18, -7, 3, -13, 9, -1, 16, 2, -10, -10, -11, -8, 11}
, {-10, 10, 1, -2, -11, 0, 6, 9, 8, 12, -2, 13, -13, 7, 11, -4, 12, 10, -5, -6, -8, 14, -15, -9, 0, 6, 9, -12, 3, 6, 9, -7, -14, -18, -1, -13, 10, -18, 4, -13, 10, -11, -18, 2, 12, 9, 3, 5, -1, -13, 14, -4, 3, -3, -14, -12, 4, 8, -13, -14, 4, 8, -4, -3}
, {-11, 8, -8, 5, 9, 16, 0, -16, 11, -13, -5, 8, 4, 7, 8, -3, -18, 2, -16, -5, 4, 13, 2, -3, 5, -2, 12, 16, -5, 10, 12, 10, -12, -18, 6, -3, -13, 4, -6, -13, 4, -17, 0, 15, -11, -3, -8, 3, -11, -3, -5, -8, -6, -6, -7, -6, -8, 7, -14, 6, 13, -11, 12, 3}
}
, {{-7, 2, -2, -9, 11, -11, -10, -14, 7, -5, -5, -9, -3, -9, -1, -3, 0, 0, 4, 7, -4, -4, 8, 10, -18, 1, 14, 8, 0, -5, -10, -9, 12, 1, -16, 9, -3, 10, -12, -6, -7, 3, 7, 9, -16, 13, 15, 10, -2, 4, -7, 0, 15, 12, -1, 13, -19, 6, -17, -6, -1, -18, -11, 4}
, {3, -16, -11, -13, -5, -7, -12, -8, 1, 15, 5, 12, 8, 4, -6, 8, -14, 11, 1, 12, -4, 9, 12, 9, 7, -17, 10, 2, 14, 5, 1, 11, 5, -6, -13, -8, -14, -16, 1, -12, -1, 7, -5, 4, -10, -16, 12, -15, 6, 3, -7, 14, 11, -8, 10, 3, -6, 11, -16, -2, 10, -14, -3, -10}
, {5, -6, 1, -8, -7, -5, 0, -10, 13, -8, 0, -11, -7, -5, 3, -6, -6, 10, -9, -15, 7, -9, 4, -7, -9, -15, -9, 4, -1, -19, -1, -4, 11, 1, -4, -8, 1, -8, 10, -2, -13, 2, 14, -2, -11, 9, 14, 4, 11, 15, -4, 13, 6, -5, -11, 6, -6, -6, 3, -11, -9, -13, 2, 1}
}
, {{-1, 13, -15, -14, -14, -10, -4, 2, 2, 9, -14, -1, 3, 5, -6, -7, -7, 6, 12, 12, -8, -9, 1, -2, 13, 13, -14, -7, 11, -13, -9, -3, 15, -7, -14, 2, 5, -14, -2, 7, 15, 14, -12, -11, 0, 2, -9, -7, 5, 0, -15, -3, 5, 5, -9, 11, -4, -7, -2, 14, -16, -5, 12, 11}
, {18, 15, 3, -1, 12, 8, -12, 3, 11, -4, -2, -6, 4, 15, 10, 7, -5, 14, -2, 13, 1, -3, 0, -6, 13, -11, -13, -8, 9, -13, 15, 12, 2, -13, -3, -6, 16, 14, -10, 7, -4, 9, -10, 4, 8, -13, -17, 8, -13, 13, 15, -17, 11, 10, 0, -6, 13, -12, 0, -13, -5, -7, -13, 18}
, {13, -1, -8, -11, -6, 9, 1, 11, 6, 17, -1, 13, 11, 14, 2, 8, -7, 2, 0, -14, -14, 15, 7, -10, -12, -4, 12, -15, 5, -11, 2, -3, 2, -3, 13, 8, 6, -2, -6, 10, -7, 5, -15, -19, 5, 1, 0, -7, 7, 12, -7, -2, -4, 5, 11, -12, -13, 15, -10, -9, 2, -11, -4, -9}
}
, {{13, -16, 6, -10, 5, -14, -2, -11, -14, -13, -1, -7, -15, -3, -15, -9, -5, 6, -17, -2, -1, -3, -1, -15, 15, 3, -7, -4, -2, 14, -13, -16, -17, -5, -18, -5, -12, -5, -4, 3, 6, -8, 3, 12, 3, 5, -2, 0, -12, 8, 10, 11, -1, -18, -11, -19, -12, 3, 3, -4, 14, 8, 6, 2}
, {-11, -4, -6, 5, -13, -14, 2, 2, -3, 6, 5, 9, 5, 12, -16, -4, 6, -15, 0, 3, 15, -7, 6, -8, 0, -1, -7, -9, 12, 8, 4, -4, 11, -5, -7, -8, 13, 12, 0, -14, -8, 12, 12, -14, -9, -2, 13, 10, 2, 0, -17, -8, -15, -15, 6, -14, 2, 2, -13, 7, -17, 6, -13, 2}
, {-3, 15, 6, -10, 4, 9, -4, 3, 8, -7, -10, -11, 0, 6, -20, -10, 8, -3, -13, 7, 2, -15, 3, -5, 4, 9, 1, -8, 15, 12, 5, -16, 10, -18, 10, -14, 7, -10, -13, 10, -4, 3, 3, -9, 9, 5, 7, -9, -7, -10, 2, -16, -4, -5, -4, -7, 0, -15, -17, -9, 3, -15, -13, -1}
}
, {{-11, 12, -6, 3, -18, 0, -4, -7, 6, -8, 5, 7, -13, 14, 5, 0, 13, -17, 9, -1, 0, -9, -12, 2, -7, -1, 11, -15, 13, -13, 3, 8, 14, 12, 11, 5, -13, -8, -5, 1, 10, -5, -12, -10, 14, -11, 15, 6, -15, 0, 4, -10, -4, 1, -5, 8, 11, 14, -3, 2, 1, 5, 8, -6}
, {8, -7, -9, 15, 9, 5, 6, -5, -3, 5, -7, -9, 3, 0, -15, 14, -15, -7, -13, 14, 8, -12, -9, -3, 12, -16, -10, 10, -1, -3, -18, 5, -8, -6, 2, 1, 1, 8, 0, 12, -14, 15, 1, -5, 0, -14, -12, 14, 2, -15, 7, 12, 7, -7, 7, 12, -3, 0, 11, 12, -9, 4, -5, 9}
, {-8, 3, -4, -11, 9, 5, 13, -3, 6, 5, 11, 3, -15, 3, -11, -6, -15, 7, -1, -5, 4, -12, 4, 1, 11, 0, 7, 8, -15, 7, -1, -10, 15, -9, -8, -5, 13, 17, -7, 9, 6, 13, 13, -9, -7, -7, -17, 12, -12, -8, -13, 4, 9, 6, -9, 4, 10, 17, 6, -9, -13, -6, -3, 4}
}
, {{2, 9, -6, 6, -7, 1, -10, -13, 12, 4, 0, 11, 7, 13, 13, 15, 9, -2, 1, -13, 0, -18, 9, 10, 9, 7, -2, -4, 6, -2, -8, -4, 2, 1, 15, 5, 7, 16, 9, 5, -10, 5, 0, -2, -6, 8, -18, 6, -3, 16, -2, -8, -8, 5, 9, 6, 9, 8, 2, -1, -2, 8, 10, -4}
, {4, -13, 11, 1, -7, -11, 7, -9, -16, 8, 4, 0, -12, -9, -16, 1, -8, -4, 9, -15, -3, 14, 13, 10, -12, 14, 6, 11, -6, 11, -12, -12, -13, -4, -12, 4, -1, -3, -16, 16, -7, -5, 3, 11, -5, -3, -10, -5, 4, -13, -9, 14, -10, 18, 1, -3, -9, -13, 11, 3, -7, -14, 16, 3}
, {-7, 5, -8, -2, 10, 3, -11, -6, 15, 11, 4, -4, 10, 6, -13, -13, 4, -15, -3, -11, -1, -5, -2, 13, 3, 6, -4, 10, -8, 9, -4, 0, -4, 8, 2, -3, -5, -6, -11, -2, -11, 13, -9, -17, 11, 3, 16, 7, 11, 8, -1, -10, 5, -8, -12, -9, -15, 9, -5, -12, -17, 13, 9, -10}
}
, {{12, 9, -13, 3, 4, 1, 4, 16, 1, 8, 14, -2, -11, -2, -5, 10, 14, 3, -5, -3, 9, -8, 12, 5, 13, 8, -3, -22, 18, -10, -6, 13, 12, 1, -4, -5, 14, 1, -14, 13, 15, -9, -9, 10, 5, -6, 9, -7, -3, 11, -13, -3, -2, -16, -16, 11, -12, -16, 7, 7, 7, -16, 7, -9}
, {11, 7, -16, 2, -8, -9, -14, 2, -7, 11, -6, 10, 14, 12, -18, -3, -7, -13, -16, -7, -17, 3, 13, -2, 12, 1, -10, -10, 9, -11, 20, -1, -7, 10, 8, -2, 6, 5, 8, 2, 13, 9, 14, -6, 10, -17, -5, -8, 0, 5, 8, -4, 5, -12, -11, -1, 16, -6, 14, 0, 5, -14, 8, -16}
, {3, 3, -7, -13, 7, 12, -5, 5, 5, 9, 3, -6, -16, -4, -1, -10, -13, 9, -5, -10, -12, -18, 16, 9, 7, 11, 7, -2, -12, 15, 14, 13, -12, -14, 6, 3, 0, -2, -9, -7, 1, -4, 8, -8, 10, 1, 7, -6, 16, -13, -9, 7, 15, 6, 14, -1, -4, 7, -7, 11, 15, 11, -5, -1}
}
, {{-1, -9, -12, 3, 7, -17, -11, -7, -14, 14, -13, -3, 12, -8, 12, -8, 1, 17, -3, -13, -13, 6, -10, 4, 3, 9, -9, 4, 2, 2, 16, 1, -1, 6, 7, 14, 15, 3, -8, -7, -5, 2, 0, -2, -3, -10, -3, 4, -1, -1, -14, -13, 3, -1, 7, -9, -3, 0, -8, -14, -13, 2, 14, -2}
, {-1, -8, 6, 1, 12, 18, 9, 0, 10, -12, 16, -10, 4, 13, 10, 11, -16, -14, 14, 5, -10, 0, 9, 17, 0, 12, 0, -5, 6, -10, -9, 12, 5, 0, -16, -4, -5, -16, 5, -5, 7, -15, 14, 13, -11, -12, -6, 11, -7, -6, -10, 4, 13, -4, -13, 8, -1, 1, 1, 10, 2, 8, -13, 1}
, {-13, 5, -13, 12, -7, -13, 8, 5, -13, -8, 11, -3, 0, -4, -10, -15, 8, 10, -3, -6, -12, -15, -2, -4, -8, -4, 10, -9, 18, -4, -13, 2, 13, -14, 11, 10, 13, 12, -5, 6, 15, -6, 6, 13, 11, 13, -12, 5, -14, 6, 12, 12, 5, 8, -8, 10, -16, -10, -14, 16, -6, 0, -2, 9}
}
, {{1, 3, -2, -14, 0, -2, 6, -16, 3, 2, -17, 5, -13, 3, 11, 13, -5, 11, -12, 8, -8, -9, -10, 9, 1, -8, -13, -6, 5, 9, -11, -16, -1, -8, 11, 3, 3, -15, -2, -18, 15, 7, -13, -7, -9, 4, 14, -7, -16, -6, -12, 11, -13, -4, -12, -3, 9, -1, -11, -17, 1, 12, 10, -13}
, {5, 16, 0, -12, -6, -1, -15, 4, -11, -3, -10, 1, -11, -14, 3, -7, -6, 13, 8, 3, -3, -8, -2, 11, 13, -9, -15, -11, 14, 8, -11, -18, 14, -4, -7, -5, -5, -1, -7, -1, -8, -8, -3, 9, 5, -10, 14, 3, 12, -11, 6, -6, 2, -1, 8, -9, 14, -13, 11, -9, -9, 0, -13, -14}
, {6, -10, -19, -11, 6, -5, 0, -3, -11, -10, -14, -10, 7, 8, -14, 5, 1, 14, -8, -7, 2, 2, -8, -1, 0, -8, -1, 0, -1, 9, 4, 0, 10, -7, -9, 6, 7, 5, -7, -10, -3, -11, 4, -6, 16, 2, -9, -5, 1, -13, 9, 13, 15, 0, -16, -6, 13, 7, -8, -14, 7, -2, 7, -8}
}
, {{-1, 5, 17, 5, 8, 6, -11, -15, 13, -12, 13, 14, -7, -3, 9, -4, 12, -20, 3, -1, 13, 4, -8, 1, -11, 1, 2, 0, 2, 7, -10, -10, 17, 5, 13, 13, -11, 13, 1, -6, -15, 12, 1, -2, 2, -5, 1, 12, 3, 11, 1, 5, 13, -7, 13, -9, -11, 5, 8, 1, -5, -13, 8, 10}
, {-11, -15, 13, 14, 8, 5, -8, 11, -16, -1, -8, 11, -5, 13, 16, -8, 1, -7, -13, 12, 5, -10, -5, 10, -11, 16, -13, -2, 14, 13, 5, -4, 4, -11, -9, -15, 3, 3, -12, -14, -14, 18, -18, -13, 7, -12, 2, -4, -2, -2, 1, 4, 6, -15, -1, 6, 21, 11, 4, 3, 9, 3, -7, -14}
, {-1, 6, 15, 5, -6, -6, -3, -2, 3, -11, -4, -4, -9, -1, 5, 0, 11, -7, 4, 10, -2, -9, -4, -5, 13, 14, 9, -1, 4, -15, 10, 9, 9, -11, 8, -6, -8, -5, 2, -11, -21, 1, 8, -14, 1, 4, -13, 1, 11, -3, 9, 14, -11, -9, -2, -9, 7, 12, -4, 2, 13, 13, -2, 10}
}
, {{16, 13, 16, -13, 12, 11, 4, -16, -8, 7, 12, -12, 9, 6, 17, 7, 7, 10, 2, 6, -1, 4, 6, 6, 0, -5, 5, 0, 13, 9, -16, 15, -6, -14, 4, 14, -7, -13, 12, -3, 4, 9, -1, 7, -17, 1, 1, -13, -16, 17, -12, 4, 11, -14, 1, 11, 17, -4, -7, -9, 7, 17, -6, 11}
, {7, -11, 7, -11, 2, -8, 12, -9, 14, 8, 8, -7, -4, -16, 14, 10, -9, -2, -1, -14, 5, -1, -14, 0, 9, 11, -9, 2, -15, 3, 9, 12, -8, 10, -9, 6, -9, 16, 4, -10, 11, 0, 11, 9, 12, 13, 14, 8, 11, 1, 6, 6, -1, 6, -16, -14, -14, 8, 1, 8, -18, 5, -12, -10}
, {7, -10, 2, 3, -5, 11, -11, -4, -7, -11, 11, -16, -6, 8, 7, -14, -15, -13, -7, -11, -15, -9, 12, 7, -4, -5, 0, -6, -1, -11, 8, 3, 1, 12, 16, 8, 4, -8, 1, 2, 11, 4, -1, 15, -2, -5, -8, 3, 6, -9, 9, -15, 14, 14, 1, -13, 1, -6, -7, -15, 8, -12, -12, -1}
}
, {{12, -4, 12, -3, -12, -1, -8, 3, 10, 12, -8, 13, -19, 11, 7, 6, 1, -1, 10, -2, 3, -15, 15, -14, 9, 4, -12, -5, 7, -10, -9, 12, 10, 2, -17, -1, 1, 1, 1, -10, -19, -13, 9, -7, 1, -3, 7, -4, 2, -2, 7, -2, -2, 13, 4, 5, 5, 4, 7, 6, 10, -14, -7, 11}
, {-13, 8, -6, 11, 8, -12, -7, 10, -7, -11, -14, 13, -14, 14, -7, -8, -13, -9, -3, -2, 11, -15, -7, 7, 3, -8, -1, 2, -2, -4, -5, -5, 9, 6, -7, 2, -13, -15, 4, 10, -13, 15, 1, -18, -4, -2, 14, -17, 2, 2, 11, 2, 0, 9, -12, 13, -11, 12, -13, -3, -4, -13, 8, -14}
, {-10, 10, -15, 8, 7, -5, -12, 11, -14, -6, 5, -10, -10, -9, -7, 13, -14, -11, -14, -4, 9, -12, -4, -9, 10, -3, -1, -14, -17, 2, -13, -4, -9, 0, 6, 6, 8, -11, -2, -12, -17, -7, 13, -3, -1, -12, 3, 9, -8, 14, 5, 10, -10, -3, 10, -16, 6, -4, -2, 13, -5, 1, -6, 3}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_273_H_
#define _BATCH_NORMALIZATION_273_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       49

typedef int16_t batch_normalization_273_output_type[49][64];

#if 0
void batch_normalization_273(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_273_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_273_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_273.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       49
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_273(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_273_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_273_bias[64] = {-41, -28, -17, 86, -63, -130, -43, -5, -6, 11, -14, -99, -106, -70, 39, -1, -59, -4, 4, -53, -65, 20, -50, -13, 27, -11, 4, 46, -34, 5, 100, 37, -31, -70, -140, -8, -1, 40, -25, 31, -73, -92, -77, -8, -70, 35, -100, -43, 31, -51, -62, 4, -3, 37, -32, 54, -44, -38, -102, -45, 56, -91, -55, 45}
;
const int16_t batch_normalization_273_kernel[64] = {153, 179, 176, 202, 185, 141, 185, 209, 173, 197, 215, 174, 194, 143, 195, 178, 202, 181, 196, 127, 117, 134, 208, 176, 145, 214, 184, 177, 206, 212, 158, 202, 199, 196, 150, 192, 157, 201, 149, 117, 175, 128, 112, 189, 118, 207, 121, 180, 133, 169, 135, 237, 172, 176, 173, 73, 190, 150, 214, 188, 104, 187, 145, 132}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_349_H_
#define _CONV1D_349_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       49
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_349_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_349(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_349_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_349.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       49
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_349(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    64
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_349_bias[CONV_FILTERS] = {-1, -1, -1, 0, -1, -1, 0, -1, 0, -1, -1, 0, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1, -1, -1, -1, 0, -1, 0, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1, 0}
;

const int16_t  conv1d_349_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{14, -16, -4, 6, -2, -3, -3, 2, -4, 11, -14, -2, 9, 6, -16, -14, -13, 3, -16, -11, -10, -15, -5, 13, -7, 2, 13, 2, -11, 3, 12, -14, 4, 4, -16, 8, -14, 2, 7, -4, 6, -15, -14, 17, -8, -6, 5, -8, -8, -4, 6, 2, 7, -17, -15, -7, -4, -13, 0, 7, 2, 12, 9, -7}
, {-8, 5, 5, -11, -6, 4, 5, 6, 9, 12, -13, 4, 12, 0, -2, -16, -3, 10, 13, 3, 12, 14, -11, 8, -5, 1, 2, 9, -6, -3, -9, 1, 14, -6, 3, 2, 9, -14, -14, -1, -1, -5, 7, -13, -6, -2, 5, 0, -6, -5, 10, -16, -3, 11, -4, 0, -2, 12, 14, 4, 16, 11, 6, 14}
, {11, 6, 11, -5, -3, 2, 7, 7, -12, 3, -15, -1, 9, 16, 5, 11, 3, -13, -8, 4, 3, -2, -11, -3, -8, -5, 4, -2, -12, -10, 9, -16, 12, 2, -1, -8, -10, -10, -5, -3, -14, -3, 1, -4, 8, -6, 13, 18, -1, 15, 3, 4, -14, -4, 1, 2, 3, 12, -16, -1, -8, 2, -14, -16}
}
, {{-1, -4, -11, 13, 9, -12, -3, -13, -15, 7, 11, 14, -13, -10, -9, -4, -17, -14, -6, 11, -6, -16, -12, -13, -1, 11, 12, -10, -8, -7, -1, -8, 11, 5, 12, 5, -14, 4, -13, -6, 2, 4, 14, -9, 15, -7, -1, -11, -10, -6, 10, -11, -13, -5, -9, -4, -14, 16, -9, 1, -19, -1, -15, 4}
, {-15, 7, 12, -12, -12, 2, 1, -9, 3, -11, -4, -4, 16, 2, 6, 9, -15, -7, -16, -1, 2, 4, -16, 17, 0, -9, 2, -12, 2, 5, -4, 1, 2, -17, -9, -1, 12, -9, 9, 10, 4, -10, -11, -13, 13, -1, -11, -3, 12, 2, 7, -15, -13, 8, -8, 2, 6, 12, -8, -13, -18, -1, -14, -4}
, {3, 17, 4, 17, 13, -2, 14, -15, 9, 14, -1, 10, -12, 9, -10, 13, -1, 3, -9, -15, 9, 5, -10, 14, -2, 2, -6, 13, 2, 15, -14, -8, 5, 11, -6, -13, 8, 2, -7, -8, -1, 13, -10, 10, 4, 6, 7, 15, -10, 1, -12, -2, 14, 10, -7, -3, 6, 11, -6, 14, -7, 5, 5, -9}
}
, {{0, 4, 8, 14, -7, 8, -9, -5, -4, 14, 7, 4, -5, -4, -5, 1, -16, 11, 3, -8, -11, 8, -8, 12, -10, 16, -8, -15, 1, 1, 7, -7, 12, -11, -12, 11, 11, 13, 2, 10, 2, 15, -2, -6, 0, 6, -5, -7, 0, 1, -11, -4, -12, 11, -5, 12, 8, 8, 0, 5, -8, -15, -5, -7}
, {12, -12, 18, -10, 12, 7, -9, 12, 2, 6, -11, -5, -6, -17, -6, 2, -9, -12, -11, -2, 2, -6, -7, 7, -17, 8, -1, 8, -13, 9, -12, 7, -13, -13, -16, 2, 13, 7, -3, 12, 16, -14, 8, 7, -10, 10, -14, -16, 9, 1, 9, -2, 9, 9, -3, -7, 5, 12, 10, 6, 7, -3, 10, -18}
, {-5, -9, 8, 2, -2, 7, -1, 6, -12, -6, -12, 3, 6, -9, -8, -1, 6, -19, 1, 11, -7, -6, 11, 8, 11, 6, -13, -1, 10, -13, 17, 3, -14, 12, 0, 3, -17, 13, -2, 3, -10, 7, -6, -4, -15, 0, -18, 3, -11, -10, 2, 1, 1, 0, 14, 14, -6, -2, -7, 16, 4, -2, -5, -16}
}
, {{-9, 10, -3, 5, 1, -11, -18, -11, 7, 2, -1, 14, -1, 9, 19, -9, 14, -1, 5, -8, 13, -4, -9, 4, 9, -12, -4, -12, -18, -9, -8, 13, -12, -17, -7, -8, -10, -5, -18, -8, -9, -3, -1, 0, -16, 6, 10, -3, 11, -2, 2, 1, -2, 5, -4, -12, 9, 12, -8, -6, -2, 11, -1, 2}
, {-18, 9, 1, 2, -4, 8, 12, 13, -7, -1, 0, -11, -9, 4, -3, 8, 4, 1, -2, 19, -9, 2, -10, 15, -4, 13, -11, -3, 3, -14, 2, -7, -8, 8, -7, -12, -2, 17, -11, -5, -4, 10, 15, 4, -15, -5, 9, 3, -2, -6, -9, -6, -6, -15, -4, 6, -12, -13, 1, 2, -12, -12, -9, -15}
, {-9, -11, -3, -2, -4, -14, -11, 4, -12, -18, -18, 13, -5, -16, 18, -6, 5, 5, 0, 7, -8, 4, 6, -13, -16, -9, -12, 9, 11, -12, 7, 5, -8, 3, -11, -11, -13, -6, 8, -6, -6, -9, -15, 13, 5, 6, 6, -2, 8, 11, 8, 9, 8, -11, -13, 9, 8, -5, 13, -10, -3, -11, 5, 9}
}
, {{8, -1, 7, 2, 7, 16, -1, 3, 16, -11, 8, -9, 8, 7, -17, -8, 10, -16, -11, 10, -8, 12, -10, -10, 13, 6, 10, 9, -12, 9, -5, 11, 19, 14, 2, -4, -2, -18, 2, 0, 5, 5, 15, -4, 12, 2, -13, 7, -8, -14, 7, 3, 0, -3, -8, 16, 12, 1, -11, -1, 14, -12, 13, 9}
, {-3, -3, 4, -14, -12, 11, -11, 4, 7, 9, 11, -11, -9, -10, 13, 12, 7, -1, -13, -12, 15, 3, 10, -13, 3, 12, -4, -4, 9, -16, -5, 8, -4, 12, 8, 2, 13, 9, 1, 12, 11, 5, -1, 13, 15, -8, 6, -2, 12, -13, 8, 3, -14, 7, -2, 0, -11, -1, 5, -9, 15, -1, -11, -13}
, {-15, -8, -14, 7, -17, -2, 10, -10, -9, -10, -7, 15, 10, -12, 7, 10, -15, -7, 5, 15, -12, -14, -3, 8, 3, 5, 11, -2, 6, -13, 3, -10, -4, 16, 12, -11, 13, -10, 12, 14, 6, -7, 5, -10, 10, -3, -14, -13, -5, -1, -3, 6, -16, -4, 3, 3, 5, -3, 15, -16, -6, 16, 10, 15}
}
, {{-9, -7, 1, -10, 3, -7, 2, -10, 10, 1, -6, -6, -15, 5, 3, 1, -14, -1, -14, 12, -2, 4, -1, 7, 0, 3, -4, 13, 4, 7, -1, -7, -3, 12, -3, 5, 6, -3, -12, 7, -2, 7, 4, 14, 2, 3, 1, -19, 18, 12, -13, 2, -10, 1, 3, 0, -2, -5, 2, -18, 11, -8, -14, 3}
, {7, -3, 14, -9, 10, 9, -15, -1, -7, 7, 4, 12, 7, -2, 3, -17, -3, -10, -14, 7, -17, -13, 2, 6, 3, 7, 5, -1, 5, -12, 16, -11, -1, -3, 0, -12, -4, 3, -14, -12, 13, -11, 8, 14, 4, 8, -6, -11, 16, -15, 1, 14, -3, 15, 4, -3, 13, -14, -13, 3, 13, -6, -7, -10}
, {-13, -14, 3, 3, 8, 2, -9, 3, 6, 0, 4, 12, -2, -1, 10, 9, 10, 8, -17, -8, -1, 13, 9, -16, -2, 3, -4, -11, 1, 15, 19, -16, 10, 8, -2, 1, 1, 0, 4, 14, 0, 9, -6, -6, 8, -12, 14, -12, 14, 11, 5, -10, 6, 19, -6, 7, 3, 9, 13, -7, 1, 5, 2, -2}
}
, {{8, -5, 8, -17, -1, -6, 13, -7, 17, 12, 5, -7, -4, -11, 19, -7, -4, 15, -10, -4, -2, 1, -13, 5, -6, 14, -11, 7, 10, -17, 17, -5, -8, -6, -3, 11, -9, 0, -17, -5, -12, 14, 4, -1, 5, -12, 4, 16, -14, -4, 0, -13, -14, -4, -5, 6, -15, -1, 0, 11, -3, -9, 16, -2}
, {-15, -15, -5, -12, 4, -14, 17, -14, 3, 10, -17, -6, -15, -6, 6, -2, 8, -11, -7, 14, 9, -2, -5, -14, 7, -1, 11, -3, -11, 8, 10, -14, 6, 8, -7, -15, -10, -4, 10, -4, 5, -15, 11, 0, 4, -12, 14, 2, 11, 8, -10, -13, 8, 18, -6, 11, -13, -10, 2, 6, 10, 3, 2, -3}
, {7, 2, 8, 6, -7, -12, -3, 1, -1, 4, -16, 9, -10, 14, 9, -9, -15, 11, -13, -15, -12, -10, 10, 13, -3, 10, 7, -1, -8, 8, -13, 5, 9, 12, 5, -4, -9, 13, 6, 0, -2, 15, 8, 6, -3, -14, -12, -12, 11, -3, -11, -11, 10, 5, -10, -7, 9, 4, 8, 4, 16, 7, 3, 13}
}
, {{1, 11, 10, 14, -8, 5, 7, -10, -8, -15, -2, 2, -15, 4, -4, -13, -14, -16, -19, 11, -8, 8, -3, -6, -2, 17, 5, -10, -3, -17, 13, -16, 3, 13, 5, -5, -4, 14, 13, -7, 7, 14, 14, 0, 6, -9, 13, -5, 1, 7, 12, -18, -14, 4, -8, 1, 8, -6, -13, 2, -6, 3, 12, 12}
, {-11, 13, -10, 3, 13, -10, -4, 11, -7, -7, 3, 13, -7, -6, -5, 3, 13, -2, -3, 13, 9, 11, 11, -9, -2, -3, -15, 2, -12, -16, -13, -5, -14, -7, 7, 4, 1, -7, 16, 8, -4, -14, -6, 0, -6, -6, -9, -3, 12, -6, -4, -3, 0, -1, -14, 7, 2, 10, 17, 4, -16, 12, 13, -6}
, {4, -16, 4, -6, -12, -1, 11, -6, 8, -13, -6, 0, -2, -13, -15, 2, 14, -7, 5, -6, 1, 1, -13, 8, 18, 17, -5, 0, 6, -14, 6, -4, -3, -8, -7, 3, -3, 3, 7, -16, 3, 3, 13, -9, 9, -8, 8, -2, -1, 2, -2, 11, 11, 13, 14, -16, -8, -14, -6, 11, 12, -10, -6, -8}
}
, {{-13, 13, 8, 1, 12, -18, -13, -1, -10, 8, -10, -18, 1, 1, 6, 9, -7, -9, -11, -12, -5, 5, -3, -1, 6, 13, -13, -11, -17, 4, -9, 9, 11, 10, -9, -11, 8, 13, -8, 9, -16, -6, -5, 9, 10, -9, -6, -4, 2, -14, 3, -11, -13, 1, -12, -1, -8, -6, 14, -5, -12, -11, -10, 11}
, {5, -17, -11, -5, 3, 5, 4, 6, -2, -8, 14, -8, -18, 8, -10, 12, 1, -16, -3, 10, 11, -2, 13, 8, 4, 4, -8, -10, -16, -15, -3, 0, -7, -3, 1, 9, -10, 6, 8, -13, -13, -4, 2, -14, 5, 4, -11, -1, -9, -15, -13, -2, 6, 8, -14, -13, 6, -12, 8, -4, -6, 9, -16, -5}
, {-4, 4, 5, -7, -11, 5, 11, -15, 3, -10, 4, 4, -15, -14, 1, -12, 10, -10, -1, 9, -4, -11, 1, 7, 5, 3, 13, 13, -13, -5, 17, -10, -3, -18, 8, 8, 3, 11, -15, -7, 14, 6, 5, 14, 3, -6, -10, 6, 7, 13, -16, -5, -12, -2, 5, 11, -19, 6, 8, -10, -9, -13, 2, -5}
}
, {{13, 9, -13, 2, 4, -3, -10, 11, -1, -10, -7, -2, 3, 9, -18, -10, 14, 2, 1, 8, -1, -2, -13, 13, 18, 15, -5, 6, 14, 11, 6, 15, 9, -9, -15, -13, 7, 11, -3, 14, 5, -6, 4, -8, 11, 2, 11, 5, -6, 2, -14, -4, -7, -5, -10, -10, -12, -12, -7, 2, -15, 4, 3, -9}
, {-16, -9, -11, -4, 14, 12, 3, -8, -13, 4, 14, -7, -9, 15, -1, 5, -14, 2, 5, -11, -4, 8, 7, 4, 15, -1, 5, -8, 5, 10, -1, 13, 0, 4, -12, 3, -2, 4, 16, -2, -9, -6, -11, 6, -7, -12, -13, -11, -9, -5, -5, 4, 0, 16, 13, 10, -16, -4, -9, -2, 15, 12, -4, -4}
, {-3, -8, -16, 1, 0, 1, 0, 9, 9, 13, 16, -10, -12, -3, -14, 6, -9, 8, -3, -12, 5, 15, -4, -11, 15, 4, 3, 12, 1, 4, 7, -12, -11, 14, -8, -12, -6, -10, 10, -5, -8, -6, -1, 5, 2, 6, -17, -14, -8, 10, 3, 11, -2, -12, 13, -11, -8, -1, 4, -15, -7, -2, 7, -4}
}
, {{-2, 1, 14, -8, -11, -14, -6, -1, -8, 14, 7, -5, -8, 9, -6, -12, -10, 5, -12, 1, 12, 15, -7, -3, 1, -16, -6, 7, 14, -4, 0, 10, 8, -8, -3, -1, -12, -4, 11, 2, 1, 9, 6, 15, -10, 10, -14, -6, -5, -3, -10, 0, -8, -11, 5, 4, -3, -10, -14, -7, 10, -5, -10, -7}
, {-12, 7, 9, -10, 16, 6, -10, -9, -12, -2, 5, 6, 9, -6, 12, -9, -9, 1, 3, 0, 6, -4, 6, 16, 6, 8, 1, -4, -7, 14, -14, 10, -9, 4, -3, -12, -10, 2, 4, 9, 12, 17, -1, 15, 13, -15, 0, -1, 8, -1, 6, -4, 2, 12, 3, 16, 1, 11, -10, -10, 0, -6, -13, -13}
, {-12, 5, -9, 15, -2, -9, 1, -6, -14, 16, 3, 15, 10, 8, -7, 13, 4, -9, 12, 8, -1, 15, 2, -3, 18, 3, -13, -12, 6, 6, 15, 13, -14, -7, -8, -6, 9, -6, -1, -17, -16, 13, 0, -3, 9, 7, 14, -14, 10, -6, -5, 9, -9, -2, -9, -11, -10, 15, 16, 11, 3, 9, 15, 3}
}
, {{-10, -7, 16, 16, -4, 8, 11, -6, -13, 6, -14, 2, 14, -14, -5, 11, -6, 13, 7, -14, -11, 3, 6, 6, 8, 1, -10, 5, -8, -1, 12, 11, 12, -15, 3, 0, -1, 9, 2, -13, 7, -1, 4, -8, -14, -15, 3, 4, -6, 0, 5, -5, 0, 12, -16, -10, -4, -14, -3, -3, -16, 5, 13, -12}
, {-11, 14, -9, -5, 4, 6, 15, -3, -9, 5, 15, 13, 8, 12, -6, -3, 1, -8, -7, 13, -3, 8, -13, 2, -2, 13, 6, -6, -8, 11, 5, -14, 14, 11, -1, -16, -1, 9, -11, 7, 9, 16, 16, 12, 6, -18, 2, -3, 8, -9, 8, -15, -9, -15, 4, 2, -2, 12, 10, 13, 1, -10, -16, -3}
, {-7, -2, 9, 14, 6, -2, 14, 11, 5, -14, 3, -4, 6, 2, 8, -1, -14, 3, 4, -2, 13, -5, -6, 17, -9, 12, 15, -4, -11, -13, -3, -9, -2, -7, 14, 2, 14, 12, -14, 13, 17, -7, -8, -12, 0, -7, -8, -14, 11, -10, 0, 3, -6, 7, 1, 4, 2, 11, 11, 15, -12, 0, -8, 15}
}
, {{-17, -15, -8, -14, 1, -2, -10, 3, 7, -4, 12, 14, -2, -4, -4, 10, -9, -7, 2, -19, -4, -9, -4, 16, 4, 4, 3, -4, -2, 7, 1, -12, -16, 11, -2, -7, -9, 10, -2, 1, 5, 14, -14, -7, -15, -4, -15, 0, -3, 12, 0, 13, 8, 16, -10, -7, 8, 0, -3, -2, 16, -13, -10, -2}
, {14, 2, -17, 3, -16, 9, 0, 11, -18, -3, -2, 3, 4, -2, -7, 14, -11, -8, 11, -7, 14, 7, 6, 15, -1, 12, -4, 5, -19, 8, -1, -3, -2, -14, 2, 3, -9, 2, -12, 9, -3, 6, -6, 8, -1, -13, 3, 10, -2, -11, -17, 13, 5, 17, -5, 13, -13, -7, 5, 5, 4, 7, 6, -12}
, {-11, -7, 5, -2, 3, 11, 2, -10, -2, 0, -16, -9, 1, 9, -9, 9, -13, -13, 1, -5, 14, -5, 8, -8, -15, -16, -9, 11, -5, 11, 7, -16, -17, -2, -6, 1, 1, 18, 6, -12, 11, -18, 12, -5, 13, 0, -15, 4, -2, -11, 5, 8, 1, 11, -10, 4, -7, 1, 7, 2, -12, 4, -7, -5}
}
, {{3, 6, -14, 0, -6, 0, 12, -15, -8, 2, -12, 5, 6, 4, -10, 15, 10, 1, -10, 5, 10, -17, 15, 5, -2, -17, 6, 14, -7, 14, 7, -12, 3, 9, -4, -8, 7, -9, -8, 6, -13, 11, 17, 6, -1, 11, 7, -2, -10, -6, -5, -14, 2, 8, -18, 11, -11, 2, 17, 0, 0, -1, -5, 4}
, {-4, -18, 8, 4, -6, 16, -3, -2, 12, -14, 11, -16, -10, -16, 19, 1, -4, -6, 2, 13, -11, -10, -9, -16, 16, -10, 12, -15, -5, 5, -3, -16, 13, -1, -14, 0, -13, 5, -13, -13, -1, -10, -7, -1, 12, 10, -1, -4, -6, 11, -2, -16, -16, -6, -3, 10, -6, 7, -6, 8, -6, -10, 7, -7}
, {-6, 9, 0, 6, -13, -6, 10, 9, 8, -9, 1, -11, 9, 10, 6, 7, -7, -6, 17, 15, -1, -4, 7, -5, -13, -8, 2, -12, 9, -1, -12, 1, -15, 3, 13, 1, 11, -13, 0, 3, 13, 12, 16, -8, -2, -9, 10, 11, -7, 9, 5, 11, 6, 6, 11, 1, -4, -8, 14, 12, -4, 16, -8, -13}
}
, {{-11, -3, -13, -4, -11, 5, -13, 2, -11, -8, -12, 1, 7, 10, 12, -6, -1, 14, -12, 15, 5, 14, -5, -2, -2, 3, -2, -14, -10, 5, -17, 2, 2, -10, 0, 12, 6, -3, -2, -7, -4, 7, 2, -6, -5, -9, 3, 1, 14, 4, -2, 13, 12, -18, 14, 3, -15, -5, -16, 11, 3, 3, 7, -4}
, {-1, 5, 10, -13, 3, 12, -3, -6, 11, -17, 7, 11, -6, -13, 18, -5, -12, 8, -9, -9, 3, -9, -15, -8, -16, -15, -12, -15, 1, 0, -18, 11, -10, 5, 7, 0, 6, 1, 4, -6, 3, -1, -13, -2, 10, -11, -7, 9, 8, -5, 1, 17, -8, -7, 8, -7, -4, -11, -3, 12, 12, 4, 9, 9}
, {10, -12, -6, -10, 1, -5, 2, -11, -6, -14, 6, 3, 11, -1, 9, -13, 17, 6, -11, 5, -6, -13, 6, -14, -2, -6, -5, -7, -13, -12, 4, -6, -16, 10, -3, 0, 3, -18, 5, 2, -16, 13, 11, 7, -15, -1, 12, 2, -1, -11, 5, 14, 15, 3, -2, -6, 4, -8, -16, -1, 12, 15, -9, 2}
}
, {{-1, 11, 7, -3, -2, -4, 9, -5, 9, 2, 2, 6, -1, -3, 4, 14, -2, -10, 3, 0, 15, -9, 11, 14, 9, -2, -8, -13, 10, -13, -12, 5, -16, -12, 1, 9, -17, 0, -2, 8, 1, 4, -16, 5, 5, -3, -7, 4, -8, 6, -17, -13, -18, -12, -11, 7, -14, 2, -12, -5, -9, -2, 1, -12}
, {10, 15, -8, 17, -11, 14, 8, 1, 8, -17, 4, -13, -6, -5, -8, 4, 1, 3, -4, 3, 10, 18, 15, 11, 11, -3, -4, -14, -6, -2, 0, 3, -8, -6, -16, -5, 0, 15, 13, -3, 8, 9, -13, 9, -9, 6, -1, 13, -7, 2, 1, -7, 11, 1, -8, 1, 8, -15, 0, 9, -12, -2, -7, 5}
, {4, 4, 2, 19, -8, -16, -2, 0, 0, -2, 12, -1, 14, -8, -12, -12, -6, 14, 4, -17, 11, 17, 2, -11, -1, 19, -5, 11, 6, 7, 2, -15, 7, -12, -14, -16, -16, 5, 1, 9, 13, -6, 2, 14, -1, 13, -3, -7, -23, -16, -2, -16, -5, 12, 10, 2, -14, -1, 7, 4, 6, -2, -10, -6}
}
, {{-16, -12, 1, -6, 7, 13, 16, 8, 10, -13, -13, 1, 12, -15, -9, 1, -5, 13, -2, -8, 6, 13, -11, 10, -12, -11, 9, 6, -8, -11, 3, -2, 1, 13, 1, -5, -3, -11, -3, -7, 5, 5, 3, -4, -13, 1, 5, -6, 2, 3, 10, -8, -14, -1, 12, -1, -7, -13, 17, -3, 4, -3, -8, -4}
, {1, -12, 10, -7, 8, 6, -3, -5, -5, -11, 6, -7, 1, 11, 8, 10, -2, -11, -14, -4, -5, -1, -1, 7, -6, -15, -5, 18, 0, 0, -14, -8, -16, -9, -6, -2, -4, 0, -11, -1, 14, 8, -4, -10, -8, 2, -6, -12, -15, 10, -10, -15, 0, -4, 0, -1, 8, -12, 12, 10, 0, -4, 1, 12}
, {-4, -7, 14, -4, 8, 0, -5, 7, -3, -4, 9, -16, -5, -11, 19, -8, 1, 12, -3, 15, 13, -2, -2, 12, -10, -9, -7, 16, -15, 7, -6, -10, -11, -12, 15, -2, 2, -10, 11, 2, -17, -17, -7, -1, 9, 13, 9, 6, -8, 11, -2, 2, 9, -10, 4, -6, 2, -6, 14, 17, 14, -13, -10, -17}
}
, {{-5, 0, -8, -6, 5, 5, 6, 5, 9, 0, 8, 5, -17, -14, 19, -1, -7, -12, 8, 1, -1, -17, 3, -13, 9, 1, -2, 7, -14, 3, 0, -3, -20, -9, 13, 10, -14, -3, -14, -12, 7, 5, -6, 9, -6, -6, 12, 10, -17, 10, -1, -7, 5, -6, 13, 2, 1, -12, 7, 4, 15, -9, -10, -13}
, {-12, -10, -9, -6, 2, 5, -1, 10, 8, -17, 10, -10, -11, -2, 4, -16, 17, 0, -12, 11, -4, 8, 4, 1, -3, -1, -17, -12, -19, 0, -9, -14, 6, 4, 14, -14, 9, 11, -13, 1, 10, 2, 5, -7, -7, 13, -15, -5, 12, 6, -11, -12, 0, 15, 12, 5, -1, -13, 3, -3, 15, -4, -9, 11}
, {4, 3, 6, -2, -7, -15, 10, -10, -9, -3, -9, -11, 10, -12, 9, 9, 2, -1, 3, -9, 6, -3, 9, 1, 0, -3, -11, 7, -12, -1, -3, 14, 10, -11, 7, -13, 3, -2, -8, 10, -16, -11, -18, -16, 0, 13, 5, 5, 6, -3, 2, 10, -12, -4, 12, 4, 2, -14, -4, -2, -5, -9, -8, -12}
}
, {{-5, -13, -4, 9, -10, 17, 10, -8, 6, -14, -16, -7, -15, 8, -2, -17, 7, -15, -9, 9, 15, 1, 14, 7, -16, -15, 10, 4, -15, 5, 7, 8, -13, 8, 1, -8, 4, -13, 5, -2, 5, -13, -8, 9, -14, -2, -2, 9, -12, -8, 15, 1, -2, 4, 2, 5, 14, 2, 13, -15, -1, 1, 13, 10}
, {-16, -3, -7, -14, -15, 16, -12, -7, 16, 13, -13, 14, -3, -15, 9, -1, 10, 10, -3, 3, -5, -8, 7, -13, 10, 11, -15, 10, 6, 8, 13, -13, 2, 0, 9, 6, 4, -3, -4, 9, -8, -7, 14, -17, 3, -8, 14, 10, -2, -7, 0, 1, 15, -15, 13, -1, -5, -6, -14, 6, 13, -7, 12, -12}
, {8, 1, -12, -1, -6, 6, -13, 14, -9, 4, -5, -1, 0, 6, -14, 10, -12, -3, -8, 2, 10, 7, -8, 13, -1, -17, 0, -7, 6, 8, -2, 11, 11, 5, -7, -6, 3, 15, -1, 15, 2, 5, -5, 14, 10, -16, -5, 10, -13, 12, 14, -9, 1, 10, 9, -2, -7, 7, -14, 1, -6, -7, 7, 2}
}
, {{-10, -5, 11, 8, 4, -2, 7, 4, -16, -2, -14, -6, 3, 3, 2, 3, 9, 7, -8, -8, 4, 5, 11, -8, 12, -10, -15, 9, 7, -10, -13, -11, 10, 13, 7, -2, 9, 0, -16, -9, 12, -12, -5, 16, 13, 4, -4, 12, 13, 0, 5, 0, -3, -5, -1, 11, 12, 4, 1, 7, -3, 9, -9, 0}
, {13, -4, -13, 10, 3, 14, -14, 5, -2, -6, 7, 13, 7, -6, 8, 9, 7, -5, 8, -10, 2, 14, -17, 7, -13, -9, 0, -13, -16, -10, -13, -12, 12, 3, -8, 5, -3, 11, -4, -4, 2, -12, 3, 6, 2, 10, 1, 5, 6, 4, 10, -10, 3, 4, 13, -7, 13, 6, 13, -12, -5, 14, -8, -2}
, {16, -1, 5, -1, 3, -14, -14, -14, 14, 16, -8, -8, 4, -5, 7, -11, -5, 3, 15, 14, -7, -8, -1, 8, 5, -14, 8, -15, -16, -9, 11, -14, 5, -8, -2, 3, -8, 12, -6, -7, 12, -13, -16, 12, -11, 2, 3, 4, 6, -8, 6, -5, -12, 8, -10, -9, -10, 12, -18, -6, 9, 15, -15, 1}
}
, {{-13, 7, -5, 2, -2, 2, 15, -10, 12, 4, -2, 4, -15, -15, 13, 5, -1, 6, -12, -15, -12, -10, -3, -14, -3, -9, -4, 10, -6, 9, -1, -5, 5, 2, -14, -4, 6, 11, -17, -6, -3, 11, -13, 2, 3, -5, -3, -1, 15, -4, -15, 13, -8, -15, 11, 16, -1, 5, -5, 4, -6, -17, -1, 5}
, {13, -15, 8, 5, -15, -2, -4, 5, 10, 9, 8, 11, -14, -3, -3, -15, 10, -10, -14, 4, 4, -11, -12, 4, 16, -13, 15, 3, -2, 9, 0, -11, -10, -7, 9, 6, -12, -10, -4, 4, 11, -16, -14, -13, -1, 10, -14, -10, 12, -7, 11, -8, 5, 9, -16, 9, -13, 12, -7, -16, 16, 1, 1, 4}
, {11, -7, 13, -19, 13, 2, -8, -7, -7, -15, 9, -4, -4, -4, 4, -15, -4, -2, 10, 5, 4, 1, 0, -7, -7, 14, 3, 12, 13, -12, 3, -12, -15, -13, -4, -16, -2, 11, -10, 16, 10, -7, -13, -9, 9, -7, 2, -8, -4, 13, -14, -2, -8, 5, -14, -5, -6, 10, -7, -11, 5, -14, -2, 5}
}
, {{13, -2, 1, -9, -3, 6, -9, -6, -3, 9, -7, 13, 9, 19, -8, 16, -7, -7, -13, -12, 0, 21, 17, 17, 15, 5, -12, 5, 0, -11, 11, -4, 2, -1, 10, -5, -10, 6, 13, 6, -11, -6, -3, -7, 1, 9, 7, -12, 11, 1, 3, -2, 16, -12, -14, 3, 3, 7, 3, -15, 1, 4, -11, 6}
, {1, -1, 13, -3, 10, -15, 12, -4, -11, 7, -10, -6, -6, -4, 11, 11, -11, 14, 1, 11, 11, -7, 12, 17, 8, 3, 8, -5, -12, -13, -17, 4, 13, -3, 9, 4, 12, 17, -5, 0, 19, -4, 11, -16, 15, 1, -3, 7, 7, 4, 6, -16, 1, -4, -9, 3, -17, -16, -4, -12, 9, 12, -4, 10}
, {9, 7, 14, -6, -10, 1, 14, 13, -17, -3, 6, -15, -16, -9, -17, 5, -13, 2, -4, 15, 2, -6, -12, -5, 5, 0, 12, -16, 2, -1, -14, -10, -3, 6, 14, 15, 5, -3, 8, 9, 6, -9, 10, 9, -7, -5, -1, 4, -10, -15, -10, 12, 2, 8, -14, 3, -16, 9, -3, 10, -10, -10, 1, -7}
}
, {{6, 3, 3, -15, -9, 12, -1, 9, 14, 8, -2, -6, -10, 10, 9, -14, 6, -15, 6, 10, -13, -15, -11, 3, 11, 1, -4, 6, -5, -12, -2, 0, -11, 0, 5, 8, -17, -17, -11, 1, -2, 15, 3, -12, 6, 6, 16, -12, -12, 12, 2, 3, -10, 10, 13, 15, -5, -7, -6, 9, -10, -20, -16, 13}
, {-13, -1, -10, -15, -10, -10, 13, -14, 3, -3, -7, 8, 10, 10, -6, 8, -11, 2, 1, -15, 9, -12, 6, 9, -8, -8, -16, 11, 1, 5, 11, -3, -14, 10, 1, 2, -16, 3, -1, -11, -2, 6, -5, 9, -11, 6, -9, -4, 10, -13, 1, 11, -13, 12, -11, -1, -12, -5, 2, 5, -1, -17, 5, -2}
, {-12, 9, -4, -14, -10, -9, -12, 3, 7, 6, 13, -1, -9, 8, 5, -8, -9, -10, 10, -3, 6, -5, 4, 12, -9, 11, 5, 2, 4, 2, -14, -13, 1, -17, -4, 14, -7, 9, 7, 3, -9, -5, 1, -10, 9, 10, -14, -3, -17, -1, 8, -10, 12, -7, -9, 12, -13, 8, -15, -8, 14, -13, 9, -8}
}
, {{10, -11, -7, 0, -15, 14, -14, 5, -9, 4, -3, -8, 1, 11, -14, -14, -10, 2, -10, 4, 2, 3, 4, 13, -14, 13, 13, 2, 4, -6, -9, 5, 8, 0, 2, 9, -2, 8, 9, -4, 13, -8, 11, 14, -9, 5, 13, 3, -9, 14, 1, -1, -3, -4, 1, 5, 14, -12, -7, -2, 8, 8, 3, -9}
, {-15, -8, 8, 4, -7, 0, -2, 6, 9, -9, 1, -2, 9, 3, -9, -16, 10, -9, -4, 10, -5, -11, 9, 17, 9, 15, -3, 0, -15, -14, 0, -15, -13, -6, 13, -5, -8, 3, 10, -1, 0, -11, -14, 4, -11, -17, 0, 13, -3, 5, -9, 15, 4, 4, -9, 18, 1, 2, -2, 4, -5, -4, 6, -6}
, {10, 6, -14, -8, 14, -11, -11, -14, 3, 17, 5, -13, 14, -12, -11, -17, 9, 14, -2, 15, 17, 10, 0, 10, -9, -9, -5, -16, 0, -16, -11, 6, 12, 15, 4, -9, -10, 9, 12, -13, 16, 2, -4, 0, 14, -13, 9, 6, 18, -2, 11, 4, -11, 1, 3, 1, 9, 14, 6, 0, 12, -7, 5, -7}
}
, {{12, 3, -16, 9, -17, 9, 3, -13, -2, -6, 0, -14, 4, -14, -2, 2, -16, -13, 3, -8, -3, -14, 0, 0, -4, 11, 4, 14, 3, 6, -12, 4, -11, 8, 7, 9, -6, -18, -2, 10, -18, -5, -13, -6, -4, 5, 2, -9, -5, -7, 0, 4, -10, -4, 11, 8, 10, -5, 9, 2, -10, -13, -13, 12}
, {-2, -15, -7, 11, -19, 10, -8, 11, 3, -4, 13, -8, 9, -7, -4, 12, -7, 4, -3, 0, 4, 0, 10, -17, 7, 2, -5, 3, 0, -15, 18, 8, -7, -10, -15, 8, -15, 7, -11, -9, -8, -4, -6, 14, 11, 9, 12, -15, 0, -15, -7, -9, 5, -11, 9, -11, 8, 13, -9, 10, 9, 0, -9, 4}
, {0, -16, 6, -5, -2, 13, 5, -13, -8, 14, -12, 2, -16, -15, -4, -5, -13, -16, 8, 6, -1, -2, -9, -7, 7, 6, -10, -2, -11, -4, -5, 6, -5, -15, 6, -2, -10, -10, -13, 0, 11, -5, 1, -8, -12, 5, -14, -14, -2, 5, 8, -7, -5, 8, -2, -3, -12, -15, -21, 14, 8, -7, -10, -14}
}
, {{-8, 3, 13, -9, -14, -3, 9, -18, -3, 5, 13, -6, 11, -18, 11, -4, -1, -14, 8, -16, 0, -16, 12, 14, -5, 1, 12, -13, 10, 6, 4, -17, 4, -12, 2, -5, 11, 5, 7, 2, 4, -4, -2, 11, 12, -13, -14, -12, -13, -11, -1, -9, 10, 11, 8, -3, 12, 6, -11, -6, -1, 6, 14, -2}
, {-15, -3, 0, -8, 1, 5, 9, -1, -2, -9, -2, -7, -12, -14, 7, -8, -12, 4, 5, 9, -2, -8, 12, 13, -13, -4, 11, -4, 4, -8, -12, -3, 9, 7, -17, -17, 3, -8, -10, -11, -7, -14, -16, -12, -6, -14, -10, -11, -11, -13, -19, 13, 10, -5, 8, 5, -15, 7, 7, -2, -7, -18, 13, 11}
, {5, 4, 5, -7, -6, 6, -10, -12, 0, -4, -16, 10, -9, -17, -16, -17, 6, -7, 7, -2, 5, -10, 9, 7, 9, 13, -1, 0, 8, 4, 12, -10, -9, 11, -7, 6, 2, -3, 10, -5, 13, 11, -3, 5, 10, 2, -11, 2, -13, -6, 8, 6, 6, -2, -5, -14, -7, -17, 7, -5, -15, 9, 13, -8}
}
, {{1, 5, 3, -6, 1, -9, -1, 8, -9, 0, 4, -11, 6, -7, -2, -8, 19, 10, 1, -1, 2, 6, 7, -2, -5, -7, -7, -3, 2, 8, 2, -7, -3, 13, -15, -15, 14, 6, 10, -9, 19, 10, 3, -12, -8, 6, 3, 10, 5, 4, -10, 16, -15, -10, 3, 7, -16, -12, -8, -10, -4, -16, -8, -7}
, {-15, 3, 12, 4, -7, 4, -13, 8, -4, 5, 3, 6, 17, 12, -1, 8, -11, 10, -12, 12, -13, 3, -12, 13, 14, 13, -1, 2, 9, -3, 15, -6, 4, 0, -13, 3, 0, 1, 1, 1, 13, 9, -7, 6, 2, 11, 7, -4, -4, 3, 1, 9, -15, 10, -3, 0, -16, 0, 0, 5, 12, -11, -2, 11}
, {3, 6, 10, -2, 9, -11, 4, 8, 4, -8, 8, 16, -13, -1, 7, 12, 9, 0, -15, -14, 3, -6, 11, -16, -10, -7, -4, -9, 7, 12, -15, -11, 6, 5, -9, 15, 9, -9, 3, 0, -9, -5, -4, 4, -9, 6, -15, -2, -2, 2, 9, 10, -14, 14, -14, -4, -14, -2, -1, 6, -11, -6, -15, 4}
}
, {{-16, 3, 0, -3, -2, 15, -11, -6, 16, 4, -15, 8, 12, 0, 2, 7, -4, -16, -11, -16, 8, 9, -4, -15, 14, 9, 13, 4, 11, 11, -12, -8, 6, -15, -11, -12, 1, -17, 9, -16, 15, -7, 8, 1, -11, -15, -13, 7, -8, 7, 0, -4, -9, -10, -5, 7, 12, 8, -14, -11, 11, 8, -1, 10}
, {15, -10, 2, 5, -11, -9, -4, -16, 14, 4, -4, 10, 4, 4, -13, -2, 7, -5, -14, 9, -14, -2, 1, -11, 13, -3, 0, -12, 4, 8, -13, -5, 10, 7, -10, 14, -16, 0, -14, 8, -1, -13, -11, -7, -5, -18, 5, 7, -12, -3, -5, 1, -2, -9, 6, -5, 5, 10, 4, -5, -5, -9, -8, -8}
, {15, -9, 11, 10, 1, 0, 4, -17, 14, 6, -2, 4, -2, 2, -20, -2, -2, 9, -16, -11, -8, 9, 8, 1, -14, 5, -12, -1, 9, 10, -1, 7, 8, 1, 3, -1, -10, 5, -4, 4, -10, 15, -8, 2, 4, 12, 6, 15, -18, 8, -8, 7, -4, -12, -4, 4, -11, -8, 1, -17, 7, 12, 15, 4}
}
, {{10, 11, 0, -15, -4, 7, 11, 13, -10, 14, 0, 6, 10, 7, 1, -15, -4, -1, -7, 14, 13, 4, -3, -14, 12, -6, -3, 5, -15, -12, 4, 5, -10, -11, -12, -14, 10, -13, -6, -13, 9, 14, 5, -5, 1, -11, -9, -10, 6, 7, -6, -2, 14, 5, -16, 8, -16, 9, 9, -5, -12, 3, -11, -2}
, {0, -14, -7, 8, -8, 2, 0, -14, 11, 10, 9, -13, 5, 7, -15, 2, 16, 3, 8, 9, -15, -16, -11, 4, 3, 9, -14, -2, 8, 11, -15, 7, -5, -2, 6, -13, 5, 8, -9, 3, -5, -3, 5, -1, -9, -5, -4, 3, 2, 3, 2, -6, -1, -6, 11, -6, -4, 1, 0, 8, -11, -7, -16, 8}
, {8, -7, -5, -3, -3, -9, -9, -10, -1, 14, 1, -11, 14, 12, -12, -14, -16, 0, -10, 14, -10, -3, 8, -7, -7, 2, -12, -8, -2, 0, 11, -12, 12, 9, 5, 11, 7, -12, -14, 2, -9, 2, -11, 8, -10, -17, 10, 1, -11, -6, -14, 8, 0, 5, 6, 13, -15, 5, 3, -4, -11, 10, 10, 1}
}
, {{12, 3, 8, -4, -9, 13, 0, -7, -6, -1, -10, 14, 2, -6, -12, 16, -4, -5, -6, 14, -9, 1, 12, 9, 5, -7, 6, 4, -10, 9, 0, 11, 10, -13, -2, -14, -11, -13, 6, 4, -14, 8, 14, 5, 8, 12, -1, 14, 5, 6, 13, -6, -9, -18, -11, -6, 6, 6, 17, -11, 9, -10, -6, -1}
, {-9, 6, -7, -9, -1, 2, 0, 5, -12, 13, 4, -12, -7, 8, -8, 1, -7, -5, 14, 0, -17, 3, 2, -8, -14, 0, -17, 9, 8, 10, 11, 14, 7, -13, 9, 8, 13, 10, -5, -16, 0, -4, -13, 9, -6, -7, 1, -8, -6, -11, 6, -18, -12, 4, 2, 13, 11, 4, 6, -2, 7, 10, -15, -3}
, {-6, 14, 9, 1, -2, 8, -2, -2, -4, 0, -17, 6, -14, -8, 13, -1, 6, -7, -3, 6, 15, 9, 5, -8, 8, 17, 1, 11, -2, -12, -8, 7, -16, 2, 2, -3, -7, -11, -6, 9, 7, 6, -16, 4, 11, 17, -7, 4, 5, -2, 6, 6, -7, 1, 7, 14, -3, -7, 3, -5, -14, 6, -16, 5}
}
, {{-1, 4, -14, 4, -3, 7, -9, 12, -12, -6, -3, -13, -12, 7, -9, 1, -4, -16, -10, -17, -9, -1, 7, 12, -5, -12, -7, -13, -8, -15, 5, 4, -8, -2, 5, 10, 14, 6, 9, -2, -17, -17, -7, 13, 2, 10, -14, 6, -16, 11, -11, -8, 1, 3, 5, -5, -16, -13, 6, 13, 10, -2, -5, -2}
, {-10, -3, -11, 16, -3, 3, -10, -1, -6, 5, -5, -11, -2, -14, -2, 15, -3, 11, -14, -15, 5, 7, 8, -3, 8, 0, 11, -9, 9, -14, -9, 5, -7, 6, 3, -2, 4, 13, 8, 13, 1, 13, -6, -11, -10, 2, 5, -1, 12, -5, -1, -18, 2, 6, -2, -15, -12, 3, -13, 6, -8, 5, 14, -4}
, {10, -14, 15, 19, 9, 3, 3, -12, 5, 7, 9, -12, 6, 12, -4, 2, 12, 0, 2, -7, 6, 6, 3, -17, 9, 8, -5, -6, -5, 10, 8, -1, -15, 4, -12, 4, 12, 15, -7, -14, 12, 5, 12, 11, -11, 17, -5, -12, 9, -11, 6, -8, 7, 7, -13, 13, 1, 11, -10, 7, -6, -10, 7, -15}
}
, {{-16, -13, 6, 6, 11, 7, 8, -17, -3, -1, 5, -1, 16, -4, -4, 15, 7, -14, 9, -10, -6, 15, 4, -1, 15, 15, -13, -9, -9, 2, 6, 11, 11, -5, 12, 0, 13, 4, 8, 4, -3, 2, 10, -5, 0, -3, 8, -10, 6, 8, -14, -10, 5, -1, -7, 7, 9, -8, 3, -8, 13, -11, -15, 5}
, {7, 4, -16, 15, 12, 2, -8, 18, 6, 9, -16, 13, -15, 0, 6, 12, -8, 1, -8, -4, -8, -7, -9, -9, 8, -7, -7, 9, 7, -9, -11, 9, 4, 15, 5, -8, 7, 2, -5, -3, -3, 5, 2, 6, 14, 7, -13, 5, 5, 7, -3, -15, 10, -12, -13, -11, 1, 12, 6, -11, 0, 10, -8, -13}
, {-2, -1, -8, 16, -7, 6, -10, 5, 3, 9, 0, 9, -16, 10, -3, 18, 8, -14, -11, 13, 12, 14, 11, 14, -9, 1, -15, 16, 0, -12, 6, 4, 9, 8, 2, 9, -4, 2, -3, 11, 7, 9, 5, -6, 16, -5, -8, -11, -16, 11, 6, -15, -10, 12, -1, -11, 10, 2, 9, -11, -9, 0, 2, -4}
}
, {{-3, -9, 17, -10, 2, -6, 13, 8, -3, 0, 0, -13, 11, 5, 5, 11, -2, 13, -4, -16, -15, 16, 10, 0, -7, -9, 11, 9, 0, -10, 0, -10, -15, -9, -11, 4, 2, -1, 9, 8, -1, -7, -4, -6, 16, -15, -15, 15, -1, 7, -5, -18, 7, 12, -3, 13, -8, 6, -16, 11, 6, -3, 7, 14}
, {12, 6, 12, -10, -12, -4, -3, 3, -6, -5, -1, -12, 13, -7, 9, -6, 6, -14, 4, 14, -8, -10, 11, 13, 18, 11, -1, 13, -4, -17, -10, 2, -11, 4, 12, 1, -4, 13, -2, 14, 15, 9, 5, -6, 7, 3, -14, 15, -11, 7, 15, -5, -17, -15, 11, -12, 9, -13, 2, -13, -15, -8, -6, -11}
, {14, -2, -12, -12, 14, 5, -11, -10, -14, 11, -11, 3, 1, 4, 0, -13, -9, -8, -14, 1, 3, 16, 11, -15, 17, 7, 10, 7, 10, -15, 9, 13, -5, 1, 5, 5, -4, 4, 9, 12, 7, 15, -7, 4, -11, -6, 7, -14, 13, 8, -12, 7, 0, -2, -3, 14, -1, -9, 15, 8, 13, 11, 16, -7}
}
, {{7, -17, -14, 0, -16, -1, 0, -16, 11, 7, -7, 10, -14, -13, 16, -12, -3, -13, 9, 3, 7, 17, -14, 10, 5, -6, 11, -3, -20, 8, 0, -11, 8, 13, 14, 1, 6, 7, -4, -4, 5, -9, -12, -13, 14, -16, 12, -17, 9, -15, 7, -18, 6, -12, -17, 5, 2, -3, 13, -3, -4, 0, 12, 16}
, {10, 7, -7, 11, 6, 4, 10, -15, -8, -3, -2, 11, -10, -3, 6, -4, -9, -7, -17, 3, -4, 10, -15, 4, -9, -18, -2, -2, -11, 11, -2, 11, -5, -10, -11, -3, -11, 0, -5, 3, -2, -7, 12, -13, -12, -10, 0, -17, 13, -7, -16, -15, -13, -6, -4, -8, -9, 15, -11, -17, 4, -4, 14, -8}
, {-14, 4, 15, -13, -1, 12, -17, -9, -7, -6, -2, -1, -14, -7, -3, 5, -6, 5, 14, -4, -17, -8, 11, -4, -4, -8, 6, 1, 7, -13, 0, -5, 0, -6, -6, 12, -13, 3, -10, 0, 9, 0, 13, 2, 4, 0, 12, 2, 9, -14, -5, -6, -7, 4, -9, 6, -5, 5, -17, -2, 3, 7, -15, -11}
}
, {{4, 4, 9, -9, 9, -7, 10, -10, 4, -12, 9, 12, 12, 7, 3, 4, 16, -12, 2, 2, -5, -7, -6, 10, -15, -9, 2, -3, -3, 6, -15, -17, 4, -12, -10, -5, -11, 8, -3, -12, 0, 17, -15, -11, -15, 13, 4, 3, 15, -9, -3, 18, 8, 10, -7, -15, -3, 11, -12, -13, -15, -6, 1, -5}
, {-8, 11, 10, -15, 0, 10, -6, 13, 12, 7, 12, 14, -2, -10, 3, -5, 0, 7, 9, 10, -11, 3, -9, -7, -7, -15, -7, 3, -2, 6, -3, 6, 13, -5, -1, -4, 12, 12, -8, 7, -2, 3, 2, 1, 2, -14, 4, -3, 14, -7, 12, 10, 11, 4, -4, -1, -11, -15, 10, 9, -1, 17, 6, -8}
, {4, -16, 3, 9, 12, 4, 0, 7, 1, 12, -8, 2, -8, 2, 15, -3, 16, 1, 4, 7, 11, -17, 4, 4, -5, -7, -6, -2, 12, 0, 13, -4, -2, 12, 15, -7, 0, 3, 12, 2, -9, 10, -8, 5, -14, -14, 7, 3, -14, -5, 13, -9, -13, 4, 5, 12, 8, 15, 0, 13, 4, -1, -5, -12}
}
, {{1, -14, 15, 7, -2, 12, -4, 6, 0, -2, -3, 5, 14, -7, -14, -13, 11, -3, -10, -3, 8, -10, -4, -7, -10, 13, 9, 9, 6, 0, 9, -12, -8, 8, 2, -10, 13, 9, 12, -11, 9, -3, 2, 9, 15, -13, -1, -10, 4, 1, -14, 7, 5, 5, -1, -1, -2, -10, -8, -1, 13, -12, -12, -15}
, {-4, 14, 0, -9, 14, -3, -3, 13, 0, -5, -2, -3, 7, -13, -16, -16, 3, -7, 5, -12, -3, 3, 10, 11, -10, -10, 8, 2, -10, 4, 14, 7, -14, -4, 14, -17, 8, 5, -4, 11, 13, 14, 15, 7, 15, 5, 9, -4, 10, -10, -10, -4, 15, -9, 12, 2, -9, -15, -4, -7, 15, 8, 5, 8}
, {7, -13, -1, -14, 9, 17, 12, 0, -14, 9, -2, 1, -12, 0, 14, -3, -4, -1, 7, -1, 9, 8, -1, -15, -10, -4, 7, -13, -2, -2, -4, 1, 15, 2, 5, -10, -7, -6, -10, -1, 13, 13, -13, 18, 1, 10, 9, 13, 0, 4, 5, -9, 3, 1, 3, -6, -5, -4, 6, 15, -1, 3, -8, 9}
}
, {{12, 3, -10, -5, 7, -1, 17, -13, -11, -14, -11, 4, 4, -7, 17, -9, -9, 5, 2, 13, 17, 10, 14, 13, -4, -1, -10, -4, -2, 14, 2, 10, -7, -2, 16, -2, 0, 8, -4, -11, 2, 4, -2, -3, -5, -7, 14, 15, 8, 18, -7, -6, 16, -15, -4, -5, -14, -7, 5, 4, 3, 0, -9, -7}
, {-3, 12, 7, 8, 15, -9, -14, -2, 13, 10, 18, 14, -11, -12, -10, 12, -4, -7, 7, 19, -3, 13, -10, -4, 15, 6, 6, -10, 6, -12, -5, -15, -13, -12, -13, 16, -11, 13, -13, 14, -9, 9, 9, -10, 8, -11, -7, -12, -3, 0, -12, -2, 12, -7, 15, -12, 3, 11, 6, -9, 1, 17, 17, 15}
, {3, 4, 11, -7, 12, 0, 12, -11, 8, 0, -10, -15, 0, 16, -2, -6, 17, 3, 10, 0, 13, -4, -8, 9, 9, -10, -13, 10, -8, -4, -10, 11, -12, -11, 9, 1, -9, -11, -6, 10, -16, -13, 5, -4, 11, 13, 12, -16, 5, 0, -14, -12, 13, -3, -7, -9, 7, -10, -1, 5, 14, 10, 10, -8}
}
, {{-6, 14, -13, 7, -10, 1, -7, -8, 11, 13, 1, 0, -7, 0, 1, -7, -9, -3, -7, 13, 8, -13, 12, 18, 14, 1, 14, 8, -4, -15, 7, 2, -4, -15, -1, -9, 9, -2, 0, 11, 10, -3, 13, 12, -7, -3, 4, -2, 3, -13, -13, -6, 12, 15, 6, -14, -17, 12, -3, -13, 14, -2, 6, -16}
, {-8, 4, 0, 13, -4, -12, 3, 12, 14, 3, -13, 10, -11, -13, -5, -11, 4, -11, -14, -12, -13, 8, -4, -8, -11, 16, -10, -3, -6, -9, -1, -16, -12, -2, 8, -4, -10, 15, -7, 8, 2, -4, 2, 3, 5, 1, -4, 12, 3, -8, -13, -6, -15, 18, -12, 7, -6, -17, -12, 6, -14, -15, -14, -14}
, {8, 1, -10, -10, -9, 9, -18, -8, -1, -11, -13, 6, -1, 4, 0, -1, -6, -14, -10, 7, -5, -1, -4, -11, 3, -3, -16, 16, -3, -11, -1, 0, 12, 8, -5, -6, 10, -13, 11, -7, 6, -11, 10, -6, 10, -5, -6, 1, -7, -8, 12, -3, 4, 10, -1, -11, -4, 12, -13, 7, 8, -2, 9, 7}
}
, {{14, 10, -16, 20, 6, -3, 2, 0, -9, -17, 1, -4, 3, 6, 7, 2, -6, -11, -16, -3, -12, -13, 9, -7, 4, 0, 14, -3, 2, 8, 14, 16, -8, 10, -11, -8, -4, 4, -5, 0, -15, 16, 1, 10, 0, -14, -4, 9, 16, 8, -12, -13, 14, -16, -8, 10, 3, 15, -15, 0, -8, 8, -16, 5}
, {-10, 0, 3, -13, -2, -1, 14, 5, 1, 0, -2, -16, -16, -10, 3, -16, -7, -4, -13, 3, 6, 16, 15, 12, 2, -12, -10, 18, 13, 9, 9, 11, 12, 5, -9, 8, -7, -11, 10, 8, -9, 8, 2, -14, 12, -10, -8, 12, 3, -8, 14, 2, -8, -7, -11, 1, 2, 11, 14, 8, 5, -2, 0, -9}
, {-13, 3, 15, -3, -3, 9, -11, -6, 15, -16, 8, 1, 15, 13, -2, -7, 5, 12, -1, 7, 3, -9, 0, -11, -5, 16, -12, -11, -5, 6, 2, 0, -1, 0, 3, -2, 3, -3, 12, -8, -4, -10, -1, -12, 10, -4, 4, -12, -3, -15, 10, 5, -5, -7, -13, 15, 5, -10, -11, -5, -1, 6, -8, -15}
}
, {{-2, 5, -14, 15, 6, -15, 7, -8, 9, 13, -3, -14, 14, -5, -8, 1, -18, -4, -3, 3, -11, -3, 16, -13, -10, 4, 13, -10, 2, -6, 10, -12, -14, -17, -2, 6, -11, 2, -9, 9, -16, 1, -10, 10, 2, 6, -6, -18, 1, 13, -15, -6, 0, -9, -16, -7, 4, -11, 11, 2, -14, -12, 13, 3}
, {-12, -10, 0, 9, -6, -14, 12, 6, 10, -9, 5, 4, -1, 6, -6, -14, -6, -15, -16, 8, -8, -14, 7, -6, -16, -9, -18, 1, -14, 11, -4, -15, 8, 10, -15, -14, -15, 5, 12, -10, -2, -3, -12, -6, -2, -11, -6, 4, 2, 2, 12, -3, -12, -1, -9, 0, 2, -9, 1, -16, 13, -13, -16, 14}
, {-3, -2, -7, -8, 10, 11, -16, -11, -5, -15, 11, 4, -13, 10, -12, -11, 10, -13, -4, -8, -1, -1, 11, -5, 12, 10, 10, 0, -7, -3, -16, -9, 13, -16, -11, 9, 0, 0, 12, 7, -5, 4, -16, -7, -3, 12, -11, 2, -14, -15, 5, -4, -16, 5, -17, -10, -15, 0, -2, -16, -17, -11, 8, -14}
}
, {{8, -9, -1, -3, 13, 7, 11, 15, -7, 10, -12, -8, -10, 0, 3, -3, -15, -6, -10, 0, 2, -4, 14, -19, 13, 13, -7, 7, -13, 9, 19, -14, 10, -7, -3, -1, 20, -4, -6, 12, -15, 9, 0, 15, -16, -9, -11, 14, 0, -1, 6, -5, 6, -9, -1, 13, 11, -12, 1, 15, -6, 4, -4, -16}
, {-8, -14, -10, -14, -11, -5, -7, 10, 11, 0, 2, 4, 9, 14, 15, 4, -8, 2, 2, -12, -14, 9, 6, -4, -3, 17, -10, 7, 5, -10, 8, -5, -8, -1, 14, 1, -5, 3, -3, -4, 13, 5, 4, 1, 9, -4, -10, -9, 9, -3, -6, -1, 3, -3, 0, -7, 2, -16, -6, 10, 0, -10, -10, -13}
, {-7, 15, -7, 11, -17, 9, -8, -14, -11, 6, -4, 1, -10, 16, -6, -2, 8, 9, -14, 3, -13, 9, 10, -1, 7, -10, 8, -1, -3, 0, 3, 14, 11, -13, 5, 6, -3, 12, 15, 17, -10, 8, 9, 5, 10, 12, 6, -13, -12, -11, -15, -9, 0, 15, -8, 3, -14, 4, -12, -8, 16, 16, -17, -8}
}
, {{-5, 1, -11, -3, 3, 3, 4, -3, 2, 13, 8, 2, -7, 9, 9, -11, -2, 16, -14, -12, 13, -12, 12, -10, 6, 3, -10, 18, 4, 9, 3, -11, -5, 15, 13, 7, -14, 1, -1, 11, 0, 3, 13, 16, 0, 5, 15, 15, 12, 11, 3, 5, -9, -5, -13, 9, 15, -5, -2, -2, -11, -1, -1, -12}
, {0, -10, -9, 10, 0, -13, -7, 4, -6, 7, 7, -8, 3, -14, 12, 14, -10, -10, 4, -12, 4, 12, 1, -3, -13, -3, 12, 1, -6, -13, -1, 0, 9, -11, -15, -11, -14, -1, 10, -5, -12, -13, 4, -8, -12, 3, -14, 14, -10, 4, 17, -15, -13, -15, -3, 3, -13, -9, -3, -13, 2, 1, -7, -9}
, {9, -16, -15, -15, -10, 9, 17, 16, -2, -13, -16, 3, -4, 12, 17, -3, -7, -12, 7, 1, 13, -8, 5, 9, -10, 3, 11, 9, -1, -5, 0, -8, 7, 4, 13, -12, 10, -15, 6, 7, -12, -3, 6, -12, -2, -15, 8, -10, 0, 18, 7, -7, 7, -8, -1, 11, 3, -6, -10, -15, 5, -10, -14, -11}
}
, {{-5, -10, 12, 10, 4, 16, -10, 14, 4, -8, -3, 14, -13, -9, -14, 7, 6, 6, -13, -11, 10, -13, -12, 20, 10, 17, 13, -4, -4, -16, 17, 11, -11, 13, -1, 2, 9, 2, 8, 0, -1, 7, 11, 7, -10, 17, 7, -14, 10, -13, -11, -11, -15, -11, -12, 8, 5, 4, 0, 12, 9, -12, -14, -7}
, {5, 3, 17, 11, 9, -1, -10, -1, -10, -8, -11, 14, -11, -1, -9, -7, -15, -7, -14, -14, 14, -13, -2, 14, -10, 14, -11, -10, 8, -15, -8, -6, 10, -14, -2, 6, -4, -14, 4, -11, -5, -3, 10, -3, -12, -8, 6, 15, -10, 15, 1, 4, -1, -13, 9, -18, 8, 5, 12, -11, 2, 2, 11, -14}
, {12, 3, 12, 15, -12, -13, -4, -7, -6, 14, -12, 0, -8, 3, -2, 8, 0, 18, -13, -15, 0, 16, -6, -5, 6, 12, 4, 13, -5, -15, 14, 14, -2, -5, 13, -4, 9, -2, 13, 12, 5, 6, 5, 1, 8, 4, 5, -1, 0, 8, 15, 2, 5, -1, 9, 3, -7, -7, 14, -14, -5, -8, 1, 5}
}
, {{-9, -5, -15, -6, 0, -5, -7, -14, 5, -14, 3, 6, -9, -1, -10, 5, -3, 13, -3, -1, 4, 3, 6, -9, 4, 0, -11, -15, 8, 11, -6, 0, -3, 6, 10, 10, -15, 7, 6, -13, -6, 4, 2, 7, -9, 14, -5, -10, 6, -14, -8, 5, 2, 1, -8, -16, -2, 1, 12, 1, 12, 12, -6, -4}
, {14, 3, 4, 7, 12, -13, 4, 11, -9, 14, 15, 1, -2, -6, -10, 4, 4, -7, 10, -6, 9, -16, -6, -9, 16, 8, 0, -18, 13, -8, -9, -2, 11, -1, -12, 7, 13, -8, 5, 12, -16, -9, 2, -1, -7, -9, 0, 0, -8, 3, -10, 17, 10, 0, -4, 4, -14, -8, 2, -12, -13, -4, 12, 14}
, {13, 1, 6, -1, 7, -13, -12, 5, -5, 13, -11, -6, 7, -11, 7, 11, 8, -10, -11, -9, -5, 13, -10, -6, -6, -6, 11, 2, 6, 0, 7, 3, -8, 1, 13, -6, 9, 2, -1, 0, 14, 2, 8, 10, -11, 5, 5, -2, 1, -8, -9, -6, 15, 10, -10, -5, -8, -18, 1, -12, -9, 0, -17, 16}
}
, {{4, 1, -14, 4, -8, 12, -1, 8, 6, 14, -1, 6, -10, -3, -13, -14, -9, -10, 6, 4, 6, 1, -9, 10, 1, 6, -9, 6, -2, 13, -14, 12, -2, -14, -12, 5, 2, -13, -2, 10, 10, -12, 5, -3, -17, 2, -3, 4, 5, -6, 5, -8, 15, -7, -11, 10, 12, 2, -1, -13, -1, 8, -8, 7}
, {-16, -3, -12, 13, -14, 7, 14, 11, 2, -14, 6, 9, -15, -11, -20, -7, -14, 6, -9, 4, -10, 2, 11, 8, -7, -15, -12, -1, 11, 15, 0, 13, -3, -1, -13, -12, 13, -13, -13, -3, -16, 13, -2, -8, -7, -12, 10, 3, -19, -8, 13, 1, -16, -16, -5, -13, -4, -11, 0, 12, -11, -2, 5, 1}
, {-2, 2, 13, -12, -17, -15, -9, 6, -17, -1, -8, 10, 13, -3, -3, -15, 0, -13, -3, -7, 10, 9, 1, -4, -17, 6, 11, -11, 3, 14, 10, 3, -5, -13, 8, 11, 0, 10, 0, 6, 3, 10, -4, 5, -6, -15, -1, 14, 11, 5, -2, 7, -10, -16, -15, -16, 9, -3, -14, 11, 7, -15, 8, -9}
}
, {{6, -16, -11, 9, -3, 5, 17, 6, 7, -10, 7, 2, 0, -12, -1, 1, 1, 3, -11, -1, 11, 12, -7, 14, 4, -9, -10, 22, 7, -13, -14, 14, -5, 7, 14, -5, -16, 1, -6, -7, -6, -1, 5, -5, -3, 10, 15, -4, -14, -4, 0, 5, -9, -12, 15, 1, -10, -8, 3, 13, 14, -6, -15, -19}
, {0, -12, -1, -16, -13, -6, 2, 1, -3, 0, 4, -10, -4, 7, 19, -13, -10, 1, 2, 9, -12, -10, -8, -15, -15, 10, -2, 16, -14, -5, 11, -2, -18, -9, -8, -14, 4, -3, 5, 12, 13, -3, -8, -3, 2, -7, -2, 5, -9, -2, 11, -14, -16, -4, -4, -8, -12, -11, 16, 18, 3, -9, 11, -15}
, {-8, -5, -4, 6, 0, -2, 2, 15, -13, -2, 10, -2, -2, -2, -5, -9, 9, 12, -16, 0, -10, 0, -11, -11, 7, 4, 2, -4, -13, -16, 2, 2, -5, -1, -10, 2, 6, 9, 13, -6, -12, -11, -5, 3, 13, -5, 13, -2, -15, 16, -3, -8, -5, 10, 7, 18, -2, 9, 16, -5, 4, -16, -12, -2}
}
, {{6, -15, -14, 9, -5, -12, -13, 10, 5, -7, -10, 14, 5, -12, 13, -6, 9, -5, -15, -14, 12, -3, 6, 4, -9, 7, 11, 15, -10, 2, 6, 4, 13, 6, 5, -4, 5, 6, -6, -12, 6, 6, -2, 9, 12, 12, 15, -14, -9, 3, 9, -1, 8, 15, 13, 10, -14, -4, 0, -13, 16, -3, 2, -15}
, {-1, -4, 2, -5, -11, 0, -16, 3, 12, -13, 1, -17, -1, 7, 8, -8, 6, 8, -5, 11, 5, -2, -3, -9, -15, -15, 4, 0, 6, -15, 6, 13, 3, -14, 1, -5, 11, -14, 5, 10, 2, -14, -5, 11, 6, 7, -9, -6, 7, -1, 8, -13, 3, 8, 2, -11, -3, -3, -16, 12, 10, 1, -10, -10}
, {6, -1, -3, 12, 10, 6, 10, -16, 8, 0, 3, 0, -5, -12, 10, -11, 0, -15, 8, -11, -4, -14, -11, -11, -3, -11, 4, 5, -13, -15, 9, -7, 0, -8, -5, 10, -10, 11, 1, 12, -6, -2, -17, 7, 14, -16, 3, -13, 5, -7, -1, -7, -5, 12, 0, 8, 4, -12, -6, -10, 11, -13, 9, 8}
}
, {{5, 1, 3, 2, -9, -13, 15, 13, 13, -11, -8, -12, 1, -11, 7, 14, -13, 5, 4, -9, 9, -2, 5, -7, -2, 0, 6, 10, 3, 2, -21, 9, 11, -7, -10, 5, 0, -14, 3, 4, -9, 1, -3, 0, -15, 13, 12, -11, 4, 10, -9, 4, -14, -3, 10, 14, 4, 0, 16, -7, 7, 9, -4, -3}
, {-13, -12, -8, -10, -6, 10, 2, 16, 10, -14, 4, 7, -4, -11, 12, -5, 0, 7, -8, -6, 11, -15, -1, 1, -15, 6, -15, 13, -14, -1, -9, -14, -16, -12, 5, 9, 11, 13, -11, -4, 2, -14, -10, -10, -9, 9, 17, -13, 10, 8, -2, -8, 8, -18, 10, 5, -6, 14, 13, 2, 14, -14, 5, -4}
, {-8, -5, 4, 3, 16, 13, 0, -1, 9, -15, -2, 6, 0, 11, 21, -9, 6, -13, 1, -2, -12, 14, -13, -3, -1, 13, -6, 18, -13, -19, -11, -18, -3, -15, 14, 5, -4, -6, 13, -18, -11, -6, 9, -7, 5, 7, 10, 14, -2, -8, -1, 0, -14, 1, 1, -7, -6, -3, 9, -5, -8, -2, -4, -11}
}
, {{-11, -9, 5, -3, 6, -6, 5, 6, 13, 14, -7, 9, 1, 6, -18, -1, -10, 5, -1, 8, -14, 12, 4, -7, -10, -15, 18, -17, 9, 3, 12, 10, 16, -16, -6, 7, 16, -6, 12, 10, -5, 11, 1, 9, -9, -2, 10, 11, 10, -11, 3, -1, -8, 6, 10, -2, 8, 12, -12, -2, 1, 6, -4, 2}
, {-5, 16, 10, 11, -11, 8, 9, -7, -6, 5, -12, -1, 13, -1, 10, 10, 7, -18, 12, -14, -12, -4, -12, 5, 2, -13, -8, -10, -2, 3, 13, -11, -9, 7, -11, -4, -15, 14, 1, -14, 9, -4, -6, 12, -8, 0, -10, -7, 8, -3, -9, 1, 10, -6, -8, -1, 3, -3, -13, -3, 6, 13, 10, 11}
, {-12, -11, -11, -3, 8, -11, 1, -5, 14, -1, 2, 7, -12, 10, 7, -14, -11, -10, -3, 1, -5, -8, 12, 0, -13, -6, 15, 0, 1, 11, -10, -7, 12, 0, 5, 11, 6, 5, 14, -5, 17, -15, 2, 12, 2, -17, -11, 4, 6, -1, 5, 15, -3, -17, -16, 8, 9, 16, -11, -7, 8, 0, 5, -9}
}
, {{-4, 0, 13, 9, -8, -5, 13, 12, -3, -16, 1, 8, -16, 15, 12, 9, -3, 13, -4, -4, 7, -12, -15, -7, 3, 0, 0, -10, 3, 17, -2, 4, 0, 2, 13, -5, -12, -14, 13, -12, 0, -5, 13, 3, -8, 0, -9, 11, -8, 4, -1, -5, -14, 2, -11, -15, -2, 15, 10, 2, -16, -10, 5, 6}
, {0, 4, -6, 6, 0, -6, -9, -12, 10, 10, -5, 15, 15, -5, -18, 0, -12, 0, 9, 8, 5, 17, 9, 10, 6, -1, 5, -8, -2, 15, 0, -7, 0, -16, -3, -11, 3, 6, 10, -8, 14, 13, -4, 9, 10, 9, 14, 13, 6, -5, -15, -4, 0, -14, 5, 7, 3, -8, -10, -10, -4, -2, 9, 14}
, {6, -1, 3, 0, -1, -14, -1, -13, 10, -18, -11, 14, 8, 10, -16, -3, 12, -9, 13, -4, 7, 10, 12, 13, 12, 4, -12, -6, 17, -13, 1, 14, -13, 4, -1, 6, -11, -15, -8, -3, 18, -15, 3, -14, -4, 10, -1, 2, -5, -2, -2, 0, -15, -8, 12, 11, -12, 1, 10, -14, -16, 13, 10, -17}
}
, {{-14, 0, 10, -6, 13, -15, -10, 2, -8, 12, 6, -13, -14, -1, 11, -1, -1, 6, 17, 1, -5, 10, -15, 3, 13, 8, -6, -2, 9, 4, -14, 9, -4, -9, -9, -6, 13, -3, 6, -17, 7, -2, -7, -1, -13, -2, 9, 13, 2, 9, 13, -8, -9, -2, -14, 13, -5, 2, -14, -14, -4, 17, 11, 8}
, {-6, -15, 2, 8, 3, 12, 5, -13, 8, -2, -1, 4, 15, -12, 2, -1, 5, 5, -7, -18, -16, -1, 1, -11, -3, -14, -6, 13, -2, 2, 8, 4, -14, -2, 3, -5, -1, 5, 1, 2, -3, -11, 7, -4, 11, -7, 2, -6, -8, 14, -14, 12, -3, -16, -10, -11, -9, 0, -9, 2, 17, 10, -5, 5}
, {13, -16, 2, -16, -9, 6, 9, -17, -15, -10, -4, 7, 7, -1, -8, -17, -15, -7, 17, -16, -6, -1, -9, 6, -10, -10, -10, -9, 9, 9, -4, 5, 13, 8, 7, -1, -7, -9, 4, 7, -6, 7, -9, 7, 13, 0, -15, 1, 15, -3, 4, 10, -7, -15, -11, 0, -7, -12, 1, -10, 3, 10, 3, 9}
}
, {{12, -2, 6, -14, -4, -9, -7, 2, -8, -2, 6, -16, -4, -5, -7, -6, 12, -10, -5, 7, 5, 0, -2, 1, -10, -13, -8, 4, -7, 7, -11, -1, 6, 11, -15, -6, -10, 5, -15, -7, 13, -5, -2, 4, -8, 16, 5, 6, 5, -2, -8, 7, 5, 12, 10, -1, 2, -2, -1, 7, 4, 14, 9, 2}
, {-8, 10, 0, 5, -15, 15, -4, 12, -16, 13, 5, 12, -12, 2, -5, -14, -11, -4, -15, 3, -12, 0, -15, 12, -4, -16, 11, 3, 7, 10, 0, 5, -11, 10, 4, -4, -3, -13, -15, 7, 14, 3, -14, 9, 11, 8, -10, 5, 1, 6, -3, -10, 6, -9, 4, -2, 12, 6, -2, -15, 6, -6, -13, -5}
, {-4, 9, 11, 8, 8, 9, -7, 11, 13, -14, 9, 6, 7, 7, -14, 0, 6, 8, -18, 16, 5, 5, -16, -3, 11, 3, -11, 10, -2, -8, -7, -5, 8, -3, 1, -1, 2, -15, 8, -2, 17, 6, -10, -4, -12, 13, 3, 15, -1, 10, -17, 4, 16, -14, -1, -1, -4, 1, -6, -15, 10, 10, 0, 4}
}
, {{-15, -7, -18, -12, 0, -16, 7, -6, -5, 6, -8, 3, 0, -8, 14, 1, -4, -7, 19, -11, -3, 7, 10, -11, 7, 5, -2, 13, 4, 3, -1, -15, -5, 10, 8, 11, -1, -11, -1, -6, 3, -9, 16, 14, -16, -5, -12, -12, 6, -5, -14, -11, -6, -5, -4, 2, 12, -1, -13, -18, 15, 6, 4, -8}
, {-14, -14, -9, -11, 6, 3, -9, 3, 7, 1, 0, 16, -16, 1, -9, 14, -5, 14, 12, 6, -5, -1, -10, -10, -15, -2, 0, 2, -2, 7, -3, -14, -14, -5, 11, 7, 9, -15, 0, 2, -16, -11, -16, 14, -5, 8, -10, 0, 11, 10, -14, -11, 1, -12, 10, 0, 1, -9, 1, 9, 9, -5, 8, -8}
, {-9, -11, -5, 4, 5, 11, 13, -13, 12, 13, -2, -15, 9, 2, 10, -14, 12, 15, 0, -6, 0, 0, 4, 4, -1, 8, -20, -2, -9, 14, -2, 14, -13, 6, 2, -10, -17, -14, -16, -6, 5, -18, -8, -14, -3, -4, 8, 13, 7, 10, 10, 17, 5, -6, 9, 6, 4, 8, -2, -7, -3, 4, -14, -15}
}
, {{3, -15, -1, -3, 3, 2, -2, -2, -2, -11, -15, 12, 10, 12, -8, -9, 4, -10, -10, -12, 7, -2, -2, 12, -9, 6, -8, 8, -3, 14, -9, -14, -12, 11, -11, -9, 4, -10, -12, -11, -1, -14, -1, -7, -17, 4, 0, 1, 9, 12, -5, -2, 13, 5, 6, 4, -12, -14, -12, -11, -10, 2, 13, 6}
, {13, 12, 2, 6, -11, -9, -9, -16, 10, -3, -10, -2, 5, 12, -19, 0, 13, 6, -2, -3, 0, 3, 3, -16, -4, -11, 0, 3, 9, -4, -16, 11, -5, 10, 9, -5, 9, -17, 0, -3, -3, -10, 0, -11, -13, -15, -5, -1, 3, -14, -13, 14, 6, -12, -13, -8, -10, 4, -11, 3, 0, -3, 7, 13}
, {-4, -4, 3, 5, 5, -13, 10, 14, 11, -15, -2, 9, 13, 0, -16, 7, -14, -14, -17, -4, -13, 10, 7, -11, -1, -19, 7, -16, -2, 1, 3, -13, -4, -2, -18, -8, 1, 6, -15, 9, -3, 4, 1, 15, -15, 7, -8, -5, 11, 12, 14, 12, 7, -3, 9, 17, 1, 9, 4, -14, 13, 9, 5, -15}
}
, {{-17, 6, -4, 8, 5, 5, 9, -4, -9, -10, -11, 15, 9, -8, -13, 10, 7, -4, -15, 5, 0, -5, -6, -5, -4, -13, 6, -9, -16, -14, 10, 3, -10, 10, -1, 12, -17, 17, 14, 1, 5, 14, -13, 12, -6, -9, -14, -4, -4, 1, -18, 8, 11, 2, 11, 8, 0, -15, 2, -12, 5, -5, -1, -1}
, {-2, 3, -5, 7, 9, 5, 9, -1, -14, -12, -14, -4, 8, 5, 2, 3, 7, -4, -2, -7, -5, 9, -4, 18, 11, -10, 5, 4, -7, -11, 8, 4, 11, 13, -16, 1, -6, 8, -2, -13, 10, 14, -1, 9, 6, 4, 5, -1, 11, -12, -14, -1, 8, -5, -14, -12, -20, 6, 13, -9, 12, 1, 3, -15}
, {-1, 7, 5, 1, -4, -4, -10, -8, 14, 5, -10, 12, -1, 4, 4, 13, -9, 10, 11, 9, -12, 9, 12, -7, -9, -9, 12, -15, 2, 12, 16, -3, 3, 15, -9, 11, 0, 8, 12, 7, -1, -8, -11, -4, -12, 9, 13, -7, 7, 11, -12, -3, -5, -5, -15, 6, 8, 5, -4, -7, -4, 9, 7, -8}
}
, {{7, 14, 5, -2, 2, 8, 9, 12, -10, 6, -9, -12, -10, -16, -9, -12, 0, 8, 3, -5, 14, -2, -1, -5, -10, 13, 3, 6, -10, -2, -12, 4, -8, -14, -5, 4, 8, -9, 11, 13, 10, -9, 6, 10, 2, 7, -11, -6, 9, -2, -15, -12, 9, -13, 16, -12, -8, 4, -14, -11, 13, -16, 12, 0}
, {13, 14, -9, 8, 14, -16, 8, -3, -6, 5, 2, 11, -11, -2, -11, 13, 13, 0, -17, -13, -5, 6, -6, 13, -2, -14, -16, -6, 0, -2, -11, -2, 10, -6, 7, 12, -1, -12, -11, 2, -10, -10, 4, -10, -4, 6, 2, 2, 14, -11, -11, -13, -11, -11, 11, 1, -3, -5, -3, -8, -1, -5, -1, 0}
, {6, -13, 14, -3, -11, -15, 12, 7, -2, 11, -3, 7, -1, 7, -7, 13, 5, -4, -8, 13, 2, -4, 4, -17, 7, -8, -3, 12, -9, 1, -14, 9, -2, 5, 7, 14, -4, -17, -8, 10, -5, 1, 14, 9, -18, 14, -10, -1, -12, -10, -19, 1, -1, 11, -7, 2, -11, -12, 12, 10, 6, -14, 12, -13}
}
, {{-1, -14, 3, -9, 7, -5, 16, -4, 11, 7, 6, -2, -4, -17, -6, -7, -12, 7, 11, 4, 7, 13, -6, -3, 3, -18, -12, -15, -1, 8, 8, -17, -6, -6, -1, 15, 5, 4, -1, 1, -8, -11, 14, 2, -19, -2, -3, 7, -17, 2, -12, 6, -7, -18, 0, 11, -10, -12, -1, 0, -3, 15, -9, 9}
, {5, -10, 7, 0, -6, -10, -14, 12, -13, 3, -3, 1, 17, -10, -11, -7, 10, -8, 15, -14, -4, -1, -15, 6, 1, -1, -15, -8, 8, -4, -17, 10, -6, -9, -7, -11, 13, 4, -2, -17, -11, -11, -8, -7, 12, 10, -7, -4, 12, 11, -4, -9, 7, -1, -3, 8, -12, -7, 2, -15, -14, -12, 1, -10}
, {-3, 0, 2, -5, 5, 14, -4, -6, 9, -9, -1, -9, 12, 1, -6, -13, 14, -2, -15, 14, 12, 0, 3, 1, -5, 10, -3, -12, 6, 0, 4, 10, 3, 14, 0, -8, -15, 3, -2, 6, -9, -1, -5, -7, -17, -13, -15, -14, -15, 4, -3, 5, 10, -6, 6, -15, -8, -14, 0, 17, -10, 7, 13, -1}
}
, {{4, -5, -4, 20, -10, 8, -15, -10, -13, 9, -16, -2, 7, 1, 13, 11, 8, 14, 12, 2, -14, -5, 13, -3, 11, 5, -15, -8, -10, -2, 9, -1, 9, -7, -14, 13, 8, 15, 12, -6, -4, -10, 13, 11, -14, 2, 3, 6, -16, -8, -5, -13, -16, -4, -5, -8, -14, 5, 3, 9, -16, 2, 10, 3}
, {4, -15, 6, -8, -13, 2, 4, 1, 15, -13, -1, 4, -4, 11, -9, -5, 0, -4, -16, -16, 7, -13, -15, 10, -10, 6, 0, -9, -10, -5, -6, 4, -8, 3, 4, -10, -18, -10, -10, 4, 6, 13, 8, 3, 11, -13, -3, 9, 14, -13, -6, -10, 13, 3, -6, -13, -15, -3, -10, 11, -7, 14, 6, 0}
, {-13, -10, 13, 12, 8, 12, 10, -7, -18, -9, -7, -13, -16, 6, -7, 14, -7, 7, -14, 9, 0, -4, -8, 4, -15, -9, -5, 11, -15, -19, 4, 8, -6, -14, 3, -13, 13, -15, -16, 2, 7, -10, -9, 2, 0, 1, 10, 3, -14, -2, -5, -6, -13, 14, 5, 10, 4, 15, 13, 1, -8, -16, 12, 0}
}
, {{1, 8, -13, -17, -11, -8, -11, -3, 0, 0, 1, -2, 2, -12, 15, -15, 14, 0, -4, -5, 13, 9, -18, 10, 4, 8, -2, -4, -10, 18, 10, 1, 15, 8, 14, 4, 0, -7, 5, -17, 8, 13, 10, 11, 8, -18, 7, -1, 17, 2, 8, -3, -1, 6, 10, 13, -10, -8, 9, -17, 6, 0, 2, 13}
, {-12, 0, 8, 2, 3, 1, 5, -2, 11, -12, -13, -3, -8, 7, -2, -6, -2, -5, -11, -1, 2, -18, 12, 4, 9, 10, 9, -1, -14, -1, 5, -8, 8, 0, 3, 10, 4, 10, -6, 9, -8, 6, 10, -9, -9, 6, -3, 1, -9, 1, 8, -1, -15, 4, -15, -10, 10, 10, 7, -15, 2, -2, 8, 8}
, {13, 7, -5, -7, 8, 16, -3, -1, -2, 18, 12, 5, 2, -1, 13, -9, -7, -9, 2, 10, 6, 10, 10, -4, 1, -5, -5, 10, -11, 10, -7, -14, -7, -3, 7, -14, 6, -3, 9, 6, -13, -8, -3, -5, -5, -10, -15, 8, 17, -5, 11, 17, -3, -11, -2, 9, 10, 12, -13, -17, -13, 0, 1, -1}
}
, {{-3, 6, -10, 5, -6, -13, 6, -13, -9, 8, -11, -7, -6, 6, -16, 3, -11, -7, 6, -7, 9, 15, -9, 3, 10, 13, 3, -10, 3, 9, -11, -6, -16, 10, -13, 12, -8, -5, -12, 14, -5, -9, 15, 0, -1, 0, -8, 1, 3, 12, -1, 9, 16, 10, -19, 1, 12, 13, 11, -10, -11, 10, -2, -14}
, {2, -3, 1, -11, -10, -1, 12, 3, -5, -12, 15, -5, 12, 3, -14, 12, -2, 10, -6, 9, 15, 11, -14, 18, -13, 5, -11, -15, -5, -18, -12, 11, -14, 11, 9, -14, -14, 3, -7, 13, 3, 12, 8, 11, 9, 6, 5, -7, -6, -3, 4, 9, -1, 0, 11, -3, -6, 6, -7, -4, -3, -1, -14, -14}
, {-11, -15, -6, 8, 6, 14, -4, 3, -7, -13, -3, -2, 16, 2, 3, 7, -7, -2, 7, 10, -10, -12, 7, -5, -6, 7, 1, -9, -7, -6, 10, -2, -13, 13, -7, -1, 3, 11, 12, -16, 13, 5, 7, -3, 14, -5, -2, 11, 1, 10, 2, 6, -3, 9, 0, 0, -10, 15, 16, -9, 2, -13, 3, 15}
}
, {{-1, -8, 12, 1, 4, 17, -6, -10, -7, 2, -5, 14, 17, 16, -10, 6, 4, -4, -1, 3, 16, 6, 17, 10, -1, -8, 4, 3, -9, -2, 7, -11, 8, -15, -9, 10, 8, 1, 15, 12, 2, 2, -10, -10, -8, -16, 18, 4, -13, 7, -6, 18, -15, -15, 6, 15, -14, -14, -14, 0, -7, -8, 10, 14}
, {-10, 18, -6, -6, 8, -1, -1, -5, -2, -3, -7, -14, 6, -5, -15, -9, 13, 6, -5, -3, 7, -1, 8, 14, 13, 11, -7, 7, -1, -7, -12, -14, -7, -12, -7, -11, 1, 14, -1, -17, 10, 11, -8, 6, 16, -6, -7, 11, -12, 19, 0, -8, -8, 2, 8, -2, -12, -14, 11, 1, 6, -11, -14, 6}
, {-7, 5, -5, -7, 12, -14, -14, 17, -3, 13, 3, 7, -3, -11, 11, 10, 0, -1, 17, -11, 17, -3, 1, 6, -5, 11, 1, -3, -7, 10, -7, 2, -10, -3, -3, 5, -16, 15, -1, -9, -13, 1, 10, 15, 16, -1, 11, -15, 8, -4, -3, 14, 8, 3, 19, 5, -10, 4, 11, -12, -4, -13, -13, 3}
}
, {{5, -18, -9, -12, -11, -9, 4, -11, 16, -1, -4, 13, 16, 4, 5, -14, -4, -10, -10, -7, 13, 0, -17, 7, 5, 6, -4, 12, -10, 2, 2, -3, 17, 6, 15, -6, -1, 4, -13, 18, 8, -11, 2, -11, 4, 12, 9, -5, -4, 7, -1, -12, 14, -10, -12, 12, 5, -1, 13, -11, -10, -3, 2, 16}
, {-1, 0, 1, -11, -6, 14, -1, -1, -3, -8, 10, -9, 16, 10, -3, 12, 2, -1, 9, -6, 3, 5, 8, 6, -5, 12, 5, -9, -15, -11, 10, -14, 3, -13, -4, 4, -12, 7, -13, 17, -17, 13, 8, 19, -3, 4, -1, 2, 3, 5, -18, 13, -11, -8, 2, -9, 14, 6, -9, -14, 3, 14, -4, 8}
, {-4, 15, -4, 8, 6, 12, 14, -4, 5, 1, 8, -8, -14, 14, 11, -8, 15, 12, 1, 3, -8, -11, -15, -16, -9, -3, -5, -6, -11, -7, 16, 13, -8, -14, -8, -14, -10, -10, 0, -9, -15, 13, -5, -8, 10, 10, 5, -9, -10, -14, 5, 5, 5, 17, 0, 5, -13, -14, -2, -5, -11, 11, -4, 6}
}
, {{-8, 14, 4, 13, 10, -2, -8, -1, 6, 14, 6, 7, 4, 2, 11, -3, -2, 16, -3, 14, 7, -12, 3, 8, -14, 2, 0, 15, -10, -9, 7, 0, -9, 14, -2, 15, -14, -10, -4, -9, 7, -3, 6, -12, 7, 4, 8, -13, -1, -11, 5, 1, 6, -11, -13, 18, 16, -1, 17, -14, -9, 1, -5, 14}
, {-8, -5, 5, 10, 7, -6, -6, -15, 5, 12, -11, 1, 10, -3, 6, 6, 4, -8, 15, -12, -3, 3, -5, 4, -8, 13, 15, -6, -6, 12, -6, -2, 8, -2, 1, 9, -5, -5, 2, 11, -8, 5, -5, -8, 7, 9, -10, -8, -2, -6, -15, 8, -9, -10, 8, -12, 2, -8, 15, 12, 2, 17, 8, -16}
, {6, 7, -12, -6, 1, 12, 14, 0, 7, -16, -9, -1, -9, 12, 11, 8, -11, 1, 3, 10, -12, 8, -12, -8, 7, 0, 14, 12, 5, 3, -11, -13, -10, 11, 16, -9, -9, 10, -15, 9, -13, 11, -4, 4, 4, 3, 12, -15, 1, -1, 3, -4, -8, 10, 13, -13, 0, 9, 1, -1, -10, -3, -12, -14}
}
, {{-8, -13, 8, -11, -8, -8, -11, -13, 13, -3, -8, 6, 9, -14, -11, 2, 7, 5, -3, -3, 4, 16, -6, -1, -5, 0, 5, -8, 12, -12, 13, 5, -14, 9, 4, -2, -1, -2, 10, 2, -7, 6, 8, 12, -4, 16, -5, -13, -13, 10, 16, -2, -14, -16, -12, 15, 0, -5, 8, 5, 6, 5, 0, 13}
, {10, 10, -6, -6, 14, -5, -9, -12, -7, -17, -2, 11, -7, 14, -7, 1, -5, -2, -6, -13, 2, -15, -8, -1, -7, 12, -11, 14, 11, -11, 7, 1, -5, 6, -1, -13, -2, 0, 14, 9, -10, -15, -2, -13, -2, -5, 9, 13, 1, 14, -1, 8, -7, 8, -8, 7, -17, 4, 13, 11, -13, 14, 12, -12}
, {-2, -9, 2, 5, -8, 7, -3, -4, -7, -4, -12, 6, 5, 15, 0, 15, 12, 7, -5, 13, 6, 14, -15, -12, 9, -15, 6, 10, -8, -9, 10, -8, 9, 3, -5, -7, -3, 7, 1, -15, -11, 12, 6, 14, 9, 3, 9, -8, -6, 1, -5, 3, -14, 2, 4, -4, -4, 11, 5, -5, -5, -3, 1, -3}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_274_H_
#define _BATCH_NORMALIZATION_274_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       47

typedef int16_t batch_normalization_274_output_type[47][64];

#if 0
void batch_normalization_274(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_274_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_274_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_274.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       47
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_274(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_274_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_274_bias[64] = {0, 40, 5, 76, -67, -85, -10, 15, 66, -30, -40, -73, 17, -37, 38, -4, 33, 33, -33, -32, 37, -42, 51, -30, 66, 35, -35, 46, 34, -35, -2, -49, -60, 87, -74, -63, -37, 64, -9, 80, -64, -2, -32, -5, 61, 71, 2, 30, -52, -24, 29, -37, 28, 67, -44, 38, 77, 76, -62, -24, -34, -11, -61, -18}
;
const int16_t batch_normalization_274_kernel[64] = {171, 186, 136, 153, 151, 203, 164, 149, 130, 181, 132, 145, 193, 141, 180, 168, 184, 133, 160, 150, 92, 146, 209, 107, 100, 121, 159, 186, 202, 173, 172, 114, 163, 142, 157, 150, 109, 150, 224, 83, 196, 177, 144, 187, 161, 151, 174, 167, 178, 101, 175, 167, 129, 143, 166, 190, 156, 203, 149, 95, 121, 187, 149, 117}
;
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_350_H_
#define _CONV1D_350_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       47
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_350_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_350(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_350_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_350.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       47
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_350(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    64
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const int16_t  conv1d_350_bias[CONV_FILTERS] = {-1, -1, 0, -1, 0, -1, -1, -1, -1, 0, -1, 0, -1, -1, 0, 0, -1, -1, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1}
;

const int16_t  conv1d_350_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0, 15, -12, -16, -15, -8, 4, -5, 4, 10, 13, 2, 11, -6, -4, -11, -6, 5, -16, -8, -10, 12, -2, -17, 14, -4, -6, -11, 8, 3, 7, 4, -16, 3, -9, -1, -17, 12, 18, 3, 6, 4, 5, 10, 21, -3, -16, 1, -2, -1, 8, -1, -7, -9, -14, -17, 5, -8, -11, -9, -1, 9, -1, 4}
, {1, 5, 6, 3, -21, 11, -1, 0, 8, -8, -2, -14, 13, -10, -12, 16, -10, -12, -4, 2, -9, -14, 7, 4, 11, -8, 3, 7, -6, -1, -12, -7, -14, -15, -16, 1, -17, 7, -4, 16, 8, -15, 17, -11, -3, 15, 0, 1, -8, -1, 14, -14, 5, -2, -10, 13, 5, 17, -1, -7, 7, 7, -19, 2}
, {17, 7, 15, 10, -11, -17, -16, -12, 13, -13, 2, -9, 3, 11, 0, 2, 1, 11, -12, 17, 3, 11, -2, -9, -14, -2, 0, -8, 6, 3, 0, 15, 13, -10, 4, 9, -11, -12, -4, 1, 9, 12, -11, 0, -7, 4, -9, 10, -1, -14, -5, -10, -3, 5, -3, 7, 10, 13, -10, -18, -7, 10, -10, -2}
}
, {{-9, 4, -10, 6, -13, 2, 7, 15, 8, -15, 13, -11, 15, -17, -3, -9, -11, 2, -15, -11, -7, 8, -14, -11, 10, -4, 17, -1, 5, -12, -4, -7, 15, -2, 0, -17, 11, -12, -10, 5, -14, 11, -15, 14, 13, 12, 1, 13, 8, 0, -6, -1, -13, -10, 11, -1, -4, -2, -3, -15, -9, -16, 13, -15}
, {-7, 3, 5, -6, 6, -4, -2, 1, -11, 9, 8, -6, -6, -12, -3, 11, -6, -18, -14, 1, -15, 0, -11, 10, -18, 11, -2, 5, -3, -1, 12, -9, -17, -1, 6, -17, 2, 0, -11, -14, 5, -16, -5, -16, -9, -1, 4, -12, 5, -8, -7, -15, 1, -14, -16, 3, 13, 12, -5, -6, -11, -3, 1, 8}
, {-3, 14, 0, -5, -4, 10, 10, 10, 7, 4, 13, -11, -4, 13, -9, -4, -12, 3, -12, 0, -13, -7, 8, 0, -14, 12, -8, -9, -3, -7, 11, 11, -14, 5, 9, 0, -4, 5, 1, 9, -13, -12, -4, -13, 12, -10, 6, 11, -11, -7, 9, 8, -12, -3, 9, -3, 6, -1, 1, 4, -7, -8, 3, 0}
}
, {{-13, -16, -12, 15, 10, 13, 16, -14, -2, 1, 6, -15, -16, 9, 0, -18, 14, -1, -13, 10, 8, 7, 4, 4, 14, 12, 15, -1, -16, 15, 3, 7, -6, 0, -9, 0, -6, -16, 12, -2, 6, 4, -1, -12, -11, -10, 9, 20, -10, -9, -3, 13, 18, 4, 1, -13, 13, 1, -12, 4, -6, 5, 7, -11}
, {7, -12, -4, -7, 8, -1, 12, -17, -13, -9, -4, -5, -1, 1, -2, -4, 9, 16, -6, 2, -5, -8, 11, -14, -7, 5, 1, 4, 5, -16, -10, 5, 4, 14, 11, 3, 7, -4, -9, -12, -12, 12, -5, -17, -9, 11, 10, 2, -3, 10, -14, 0, 5, 2, -14, -14, 6, 9, 13, 9, -8, -14, -4, 10}
, {-4, -17, 11, 14, -18, -2, -4, -2, -8, -11, -18, -4, -9, 13, 14, -12, 5, -4, 1, -9, -12, 2, -3, -19, 9, -7, -5, -13, -9, 11, -2, 6, 6, 6, -9, -5, 0, -7, -3, -4, 4, -12, -10, -15, 5, -1, 1, -6, 7, -11, -4, 4, 11, 8, -2, -3, -7, 11, -3, 11, -17, 4, -2, 11}
}
, {{-1, -5, -11, -5, 6, 6, -10, -4, 7, 2, 1, -10, -8, -9, -13, 15, -3, -4, 13, -14, 0, 10, -2, -4, -14, 11, -18, 13, -12, -6, 5, -3, 9, 6, -6, 2, -11, -10, 5, -15, -3, -15, -18, 4, 3, 5, -10, 15, 5, -8, 6, 12, -9, 9, -9, 1, -6, -9, 13, -12, 6, -4, -16, -14}
, {16, -4, 9, 13, -15, -10, 9, 0, -14, -9, -9, -12, 4, 0, -9, 4, 15, -7, 15, -14, -13, 2, -3, 14, -7, -6, -10, -12, -5, 2, -8, -15, -8, -13, 6, -1, 11, -10, -1, -12, 13, 9, 7, -11, -2, -2, 7, 4, 2, 1, -18, 7, -1, -20, 8, -16, 4, 14, -12, -9, -11, 15, 6, -8}
, {-13, -4, 7, -3, -3, 10, -11, -2, -5, 8, -5, -6, 10, -5, 6, -7, 9, 3, 5, -10, -8, -8, 2, 2, -2, -6, -17, 3, -11, -9, -7, -18, -9, 7, -6, -8, -6, -5, 4, 14, -10, 10, -15, -7, -14, 11, 15, -6, -8, 5, 10, -15, -11, -16, -3, 0, 12, 2, 3, -6, 4, 15, -13, 7}
}
, {{11, -10, 5, -17, 0, -13, 12, 17, -2, 5, 11, 9, 2, -9, -6, 14, -7, 1, -4, 9, -15, -4, -2, 2, 12, -2, -14, -14, 16, 9, -1, 14, -13, -11, 9, -15, 0, 9, 3, 7, -19, -1, 0, 3, -8, -9, 11, -4, -12, -14, -14, -7, -12, 13, -11, -6, 13, -14, -13, -7, -11, 11, 13, -1}
, {10, 10, 5, -12, 12, -8, 3, 11, 5, 13, -8, -12, -4, -11, 7, -9, -4, -15, 0, 10, -8, 12, 10, -8, 3, -10, 14, 0, -12, 1, -6, -1, -14, 10, -5, -14, -15, 13, -3, 3, 1, 1, 5, 4, -5, 4, 1, 14, -1, 11, -2, -17, -3, 3, 12, 14, 13, -9, -16, 0, 14, -15, -6, -7}
, {-14, -14, 9, -4, 11, -8, 4, 0, 7, 6, 4, -7, 15, -8, -10, 15, -5, -16, 5, 14, 10, 8, -3, 5, 7, -13, 14, -1, -13, 7, -15, -12, -9, 2, -19, 3, -12, -18, -4, 11, -14, 1, 6, 15, -14, -18, -1, -8, -13, -10, -5, -13, 7, -13, 8, 6, 9, -7, 4, -6, -7, 9, -2, -18}
}
, {{-5, -15, -9, -6, -16, 13, -10, 9, -15, -6, -2, -9, -15, -9, -3, -12, 18, 3, -2, 12, 14, -7, -9, -6, 6, -7, 2, -19, -12, -17, -4, 4, 5, -12, 5, -1, 11, -6, 7, -8, -9, -3, 10, -19, -17, 9, 9, 18, 8, 6, -1, 4, 2, 9, -1, -20, -9, 16, -9, -12, -17, 3, 9, 7}
, {-9, 0, 7, 12, -9, -2, 16, 2, -8, 4, -17, -8, 9, 7, -15, 3, 15, 0, 5, -9, -11, -14, 12, 4, 6, 9, -7, -8, 12, 3, -8, 3, 4, 4, -16, 4, -2, 0, -6, 11, -4, 3, 11, -2, 6, 3, 6, 8, 10, 0, -17, -5, 7, 9, 0, -16, -13, 14, 9, 5, -4, 1, 9, 12}
, {-15, 8, 7, 17, -11, -9, 10, -5, -9, -15, -11, -14, -7, -13, -9, 7, -1, 15, 13, -3, -7, 10, -14, -1, 11, -7, -2, -19, 3, 10, -3, -7, 1, 5, -6, -4, 0, -12, 7, -1, -1, 6, 7, -11, -9, 18, 6, 1, 0, -6, 8, 6, -5, 2, 8, 7, 0, 0, 0, 8, -17, 0, -1, 3}
}
, {{4, -8, 9, 13, -10, -4, -4, 8, -2, -14, -3, -6, 10, 15, 6, -2, 4, -1, -8, 13, 8, -10, 6, -7, -6, 7, 4, -14, -6, -14, -12, -4, -8, -2, 13, -4, -14, 5, 6, -11, -10, -15, -14, -6, 11, -8, 2, 5, -9, -2, 1, -9, 12, -14, 0, -16, 10, 12, -1, 2, 1, -5, 14, 6}
, {-6, 12, -11, 2, -1, 5, 10, 13, -5, 5, 13, -9, -6, 2, -15, 12, -15, 3, -9, -8, -12, 9, -4, -13, 6, 14, 1, 11, -15, 2, -3, 2, 4, -14, 8, 13, -10, 11, -15, 2, 3, -2, 9, 14, 11, 2, 2, -15, 2, 14, -12, 0, 5, 2, 8, -14, -8, -15, -1, 10, 4, -6, 1, 13}
, {10, 2, -4, -6, -3, 13, -15, 13, 16, -7, -13, -11, 13, 10, 6, 3, 5, -3, -2, -7, -1, -16, -8, -15, -2, 15, 0, -7, -14, -7, 12, 8, -11, 8, -5, -13, 13, 7, -15, 11, 12, -3, 3, -14, 2, 2, -3, 11, 3, 4, -18, 6, -8, -12, 1, -3, 3, -6, -12, 5, -3, -5, -4, -14}
}
, {{-14, -16, -3, 3, -3, 3, 2, 13, 14, -9, -4, -17, 9, -10, -7, 8, 17, 7, -12, 9, 9, -13, -13, -3, -4, 0, -11, -4, 5, -3, -7, 6, -4, -6, 4, -6, 13, -2, 0, -9, -5, 20, 1, 1, -21, 23, 13, -6, 6, -4, 8, -14, 8, -17, 2, -6, -16, 3, 11, 7, -9, 7, -3, 10}
, {-2, 1, 9, 9, -14, 12, -7, 2, 10, -7, 0, -17, 0, 1, 6, 1, 1, 12, -13, -13, -6, -8, -14, -2, -15, 11, -2, 0, 3, 5, -1, -10, 5, -5, -7, 7, 6, -9, 10, 7, 12, -5, -16, 5, 5, 17, -9, -5, 5, -4, -1, 7, -2, 12, -10, 6, 2, 17, -18, -10, -6, -11, -3, -8}
, {10, 2, 1, 9, -2, 7, -1, 10, -1, -14, -13, -7, -8, -2, 9, -13, 7, 19, -5, -7, -14, -8, -4, -2, 6, -9, 10, 9, -6, 12, -6, 6, -14, 15, -16, 4, -3, 3, -3, 6, -1, -12, -6, -17, 10, 22, -13, 17, -19, 13, -3, 4, 1, -7, 5, 10, -9, 0, -5, 11, 5, 10, -2, 10}
}
, {{4, 3, -20, 10, -9, 1, -11, -1, 3, 8, -10, -12, 2, 12, -2, -18, -8, 22, -13, -2, -14, 2, -18, 15, 1, 6, -7, 6, 12, 7, 7, -12, 12, 11, 7, -2, -12, -14, -1, -19, -6, 12, 11, -6, 9, -6, 13, 13, 15, -4, -2, 6, 18, -11, -6, -3, -7, 4, -7, -3, 6, -9, -4, 11}
, {13, -13, -17, 9, 11, -13, 10, -16, -19, 12, 16, -11, -18, -6, -7, 5, 2, -1, 7, 14, 11, 0, -2, -1, -11, 5, -2, 3, -11, -2, -5, -9, -2, 0, -11, -13, -16, -5, -7, -4, 12, 1, 11, -9, -14, -8, -3, 19, -4, -17, -7, -10, 23, -12, -4, -12, -2, -10, -13, 8, 0, 15, -7, -11}
, {-9, 4, 0, 9, 4, -16, -7, 0, -22, -11, -9, 1, -17, 3, 1, 6, 7, 17, -8, 8, -7, -7, 8, 3, 1, -2, -8, -10, 2, -9, -11, 12, -2, -14, 1, 2, -4, -5, 3, 0, -14, 15, -15, -18, 5, 6, -5, 21, -17, 3, 17, 7, 19, -1, -9, -14, 0, 12, -12, 2, -8, 0, 15, -2}
}
, {{-6, 6, 13, -11, 0, -9, 8, -2, 2, -13, 13, 8, -9, 12, -1, 6, 3, 11, -6, 8, 11, -4, 14, -5, 7, -8, 0, -16, -20, 13, 10, -8, -8, 2, -1, -3, -16, 14, 7, -14, 7, -8, -13, 5, 9, -11, 13, 13, -1, -6, -5, 8, -11, -13, -13, 1, -4, 16, 0, 5, 13, -5, 10, -14}
, {10, 14, 5, 13, -17, -6, 0, 11, -1, -10, 6, -7, 3, -5, 2, -7, -8, 12, -18, -13, -9, 10, 2, 8, -14, -7, -2, -18, 7, -2, 12, -3, 14, -15, -12, -1, -11, -12, 12, -12, -5, -6, 12, 0, -9, 5, -9, -12, -10, -6, -15, 0, -5, -2, 5, -6, -12, -6, -9, 13, -8, 7, 14, -7}
, {0, -12, 0, -4, 6, 3, -5, 13, -7, -8, -1, -2, -7, 13, -8, 1, 14, -12, -10, -15, 1, -4, 12, -17, -1, -15, -10, -10, -5, 3, 17, -14, 13, -13, 12, -12, 2, 15, 14, 13, 10, -10, 11, -13, -6, 3, -2, 3, -15, 6, -9, -15, 6, 4, 1, -11, 12, 11, 11, 9, -2, -12, -6, 12}
}
, {{-7, -12, 13, 4, -15, -19, 12, 14, -7, -5, -6, 3, -10, -10, 2, -7, 3, -5, -8, -4, 3, -2, -2, 7, 7, 11, -4, -10, 6, 11, 11, 14, -14, -2, -12, -7, 13, 4, -8, -5, -16, -10, -5, 11, 0, 8, -11, -11, 5, 10, 0, 10, -15, -13, 4, -5, 12, -11, 11, -7, -10, 10, 6, 7}
, {-5, 2, 12, -2, 1, -3, 15, -14, 8, -11, -6, 5, -3, -10, 8, 13, -14, 1, -12, 8, -14, -6, -11, -1, -6, 9, -10, -3, 0, -9, -12, -4, 9, -10, -15, -8, -10, 8, -2, -3, -15, -1, -1, -1, 12, -6, 2, 8, -1, 2, 7, -5, -10, 4, -12, 11, -1, 3, -2, 14, -5, 12, -1, -18}
, {15, 6, 8, 3, -17, 16, -9, 9, -3, 1, -11, -14, 10, 5, -7, 12, 6, 12, -3, -2, -13, -2, 7, 0, 11, 5, 14, -9, -9, -10, -5, -4, 7, 4, -7, -1, -6, 12, -7, 5, 1, 3, 12, -4, 11, -12, -4, 7, 12, -7, -3, 3, -4, 8, 6, 5, -7, -5, 9, 5, 2, -16, -16, 5}
}
, {{-5, 11, -7, -11, -7, 4, -11, -15, 1, 3, -3, 8, -9, -11, -2, 13, 5, -15, -1, -12, -15, -6, 11, 15, -8, 11, 9, -17, -16, -17, -3, 4, -17, -7, 5, -4, -1, -6, -1, -9, 16, -6, -3, 16, 9, 3, 3, -13, 13, -4, 10, 4, 1, -13, 9, -2, 7, 9, -3, 0, -10, -4, 12, 11}
, {6, -7, 14, 11, -12, 8, 9, -4, 12, 13, -11, 1, -9, -9, -5, 11, 13, 8, -8, -12, 12, -11, -6, -13, 14, -7, 11, 1, 5, 11, -2, -5, 4, -17, -6, 14, -3, 10, 9, -14, -6, 4, 16, -4, 6, -5, -1, -4, 1, 12, -9, 16, -11, 9, 14, -11, -7, 6, -13, 8, 13, -16, -8, 1}
, {5, 11, 5, 13, -14, 14, -8, -12, -11, -3, -8, -9, -3, -12, -2, 16, -16, 11, 8, 3, -2, -3, -16, -11, -5, -3, 11, 13, -5, -6, -1, 4, -1, 3, 1, 3, -4, 9, 1, -10, -16, -7, 12, -8, 11, 7, 9, 10, 10, 9, -14, 13, 6, 8, 12, 12, -7, 15, -16, -15, 13, -10, 2, -2}
}
, {{-6, -11, -5, -14, -11, -2, 1, -11, -9, -2, 16, 16, -14, 10, 11, -1, -11, 3, -7, -12, 14, 2, -18, -7, -5, -4, -11, 22, -6, 9, 3, 7, 12, -12, 7, -4, -13, 11, 7, -9, 9, 7, 0, -15, -9, 12, 4, -5, 19, 0, 6, 13, 12, -12, 13, -3, 3, -3, 6, -2, -16, 11, -15, 1}
, {0, -17, -15, -15, 14, -13, -3, 14, 12, 13, -9, 13, -7, 11, -16, -4, -19, 8, 7, 13, 13, 3, -3, -1, 4, -11, -1, 15, 8, -9, -9, 9, -7, 0, 8, -1, 4, -2, 13, 6, -5, 7, 8, 10, 15, -11, 9, 7, 20, -17, -5, -9, -1, 1, 10, -1, 7, 5, 3, 6, 7, 10, -4, -17}
, {-11, 14, -10, -3, 4, -15, -7, 13, -10, 4, 11, -10, -3, -3, -7, -9, -2, -11, -8, -13, -6, 7, 2, -1, 2, 5, 3, 16, 4, 12, 13, -1, -13, 11, -5, 1, -8, 9, 5, -10, -16, -9, -11, 17, 17, 4, -1, 5, -6, 3, -10, 5, 11, 14, -17, 2, -8, 9, -8, 11, -13, 11, 1, -5}
}
, {{11, 0, -2, 9, -10, -15, 2, -8, 10, -9, -1, -10, 15, -8, -7, 1, -4, 11, -10, 1, -6, 3, 0, -5, 10, -2, -13, -12, -16, 10, 11, -6, -6, 9, 12, 7, -15, 8, -10, -13, -5, -3, -6, -1, 9, -12, -5, -12, 8, 8, 1, -11, -20, 13, -5, 8, -3, 9, -6, -11, 12, 1, 12, -3}
, {5, -6, 6, 15, -4, -2, -1, -2, 8, 9, -6, 6, -2, 3, 11, 0, -15, 10, 0, 11, -14, 11, 9, -11, -8, -3, -14, 0, -3, 10, 3, -5, -17, -7, -9, 1, 13, 9, -9, 9, 2, -12, -6, -7, 10, 16, 14, 6, 16, 5, 2, 14, -9, 9, 6, 12, 4, 11, -17, 9, -16, 10, -12, -8}
, {4, 6, 12, 2, -3, 11, 12, -2, -14, -15, 0, 4, -13, 1, 0, -7, -17, -4, 6, 4, 15, -15, 5, -4, 2, 4, 0, -12, 13, -19, -16, -1, 5, -3, 12, 2, -15, -14, -3, 13, -5, -1, -13, -3, 8, -4, -7, 0, 12, -17, -17, -17, 11, 3, 3, -9, -10, -12, -3, -7, 12, -10, -1, -4}
}
, {{-10, 3, -12, 0, -7, -14, -6, 7, 3, 6, -8, -3, -12, -7, 9, -4, -15, -8, -18, 10, 0, -2, -1, -16, 2, -4, -12, 13, -9, -17, 15, -5, 8, -17, 6, -7, -13, -2, 11, -13, -1, -6, 14, 15, 0, -6, -15, -1, 13, -2, 13, -5, 8, 3, 12, 2, 11, -12, -5, -1, 2, 12, -6, -13}
, {10, 1, 5, 3, 6, 1, 11, -4, 8, 3, -15, -3, 10, -8, -2, 12, -6, -3, -15, 0, 2, 16, -13, -17, 3, 13, 8, -12, -3, -2, 4, -2, -14, 4, -9, -8, -2, -11, -1, 13, -12, 2, 14, 9, -6, -15, -9, 9, -11, 8, 0, -2, -13, -15, 5, 11, 10, -10, -1, -10, -8, -15, 6, -14}
, {9, -9, 5, -5, -13, -3, -3, -5, 6, -5, -14, -7, -8, -6, -13, -1, -9, 4, 5, 8, 11, -11, -12, 2, -19, 16, 16, -2, 1, -16, 8, -16, 8, -8, 14, -1, 15, 6, -16, -1, 5, 13, 8, 3, 3, -3, -5, 6, 3, 8, -14, 7, 7, -15, 1, 15, -3, -3, -13, 3, -13, 11, -7, -5}
}
, {{-9, -7, 12, 1, -15, -9, 5, 9, 6, -13, 6, -12, 4, -16, -8, 15, 4, -8, -9, -3, 1, 9, 12, -10, -6, 2, 16, 10, 8, 6, 17, -12, 13, 5, -13, -9, 3, 10, 6, 9, -10, 8, -5, 16, -14, -4, -8, 16, -10, 12, 8, 11, 1, -12, 9, -2, 5, 0, -3, -11, -4, 15, -4, -12}
, {7, 4, -9, -9, -5, -12, 1, 17, 12, 1, -1, 7, -7, -7, 8, -6, 4, 11, -11, -4, 3, 14, -12, -8, -5, 15, -10, 11, 4, -6, 1, 16, 6, 5, -18, -7, -5, -3, -13, -7, -2, -8, -6, 16, -6, -17, 2, 1, -6, 1, -17, -16, 7, -14, 0, 18, -8, 1, -2, 3, -6, -12, 4, 5}
, {-19, -8, 18, 8, -2, 14, -15, 9, 14, 10, 2, -2, 2, 0, 11, 14, -7, -11, 0, -4, -8, 16, -13, -2, -4, -13, -7, -4, 4, -6, 10, 9, -8, -9, 10, 4, -7, -17, -19, 10, 4, 11, -6, -6, 6, 3, -1, -12, 3, -1, 2, 6, 11, 2, -5, -9, 13, 6, 13, 3, -6, -1, 1, 6}
}
, {{-1, -5, 7, 12, 10, 6, -11, 6, 6, 3, -8, 15, -5, 2, -15, 18, 10, -10, 7, 11, -14, 11, 17, -13, -2, -4, -2, -16, -8, 0, 4, 1, 15, 14, 11, 9, -15, 14, -4, 1, 6, -13, 16, -11, 11, -10, -11, 4, 0, 0, -6, 1, -1, -1, 13, -12, -3, 3, 1, 10, 0, -14, -10, 9}
, {-8, 6, 5, 1, -2, 13, -15, -4, -4, 2, 1, -4, -3, -6, -13, 9, 2, 5, 7, 1, -5, -2, -11, -10, -10, -5, 1, -15, -12, -17, 3, -5, -10, 13, -15, 8, -10, 5, -10, -4, -4, -2, 3, 3, 2, 4, -16, -2, -1, -4, -14, -12, 3, 2, 7, 6, 8, 0, 14, 0, 2, 10, -5, -10}
, {7, 13, 5, -4, 5, 7, -10, 13, 20, -2, -2, 6, 0, -6, 7, 6, 14, -14, -13, -9, -1, -15, 18, 11, 8, -4, -7, 8, -15, 2, -13, -11, 12, 10, 9, 6, 3, 3, -4, 5, 16, -5, 7, 1, -8, 10, -7, -2, 3, 4, -14, -9, -6, 10, 4, 16, 9, 14, 7, -7, -13, -12, -8, 3}
}
, {{13, 8, -11, 15, -16, 12, -4, 15, -8, 11, 6, 11, 4, 12, 4, -8, 7, 1, 0, -17, 10, 14, 5, 10, -4, 5, -2, -12, -18, -9, 18, -15, -15, 7, 3, -3, 9, 6, -8, 10, 10, -2, 16, 13, -9, -3, 7, 15, -1, -7, 1, -5, 0, -7, 14, 9, -6, 7, -14, -1, 2, 1, 6, 6}
, {-14, -2, -3, -2, -16, -7, 10, -14, -12, -3, 0, 2, -3, -4, -14, -1, -7, 14, -10, -8, 13, 11, 15, -15, 7, -6, 4, -5, -2, 3, -11, -11, -2, -4, 3, 7, -14, 11, -6, 1, 0, 5, 3, 7, -17, 5, -13, 16, -10, -6, 7, 7, 7, -13, -16, 11, -9, 11, -2, -5, -2, -15, 11, -4}
, {5, -5, -9, -11, -11, 6, -3, -7, 12, -2, -8, 5, -10, -1, -6, 16, -6, -3, -16, -8, 8, -13, 4, -18, 12, -15, -1, 11, -6, 9, -8, -1, 13, 4, -19, 4, 8, -1, -11, -4, 5, 4, 13, -13, -18, 1, 15, -10, -7, 9, 7, 9, -6, -16, -16, 2, -3, 7, -1, -17, -6, 9, 7, 6}
}
, {{8, -11, -4, 12, -6, 5, -8, -2, 13, -15, 12, -7, 4, -4, 7, 4, 5, 9, -16, -18, -16, 6, 14, 6, -10, 1, 0, -8, -5, 5, 12, 8, 5, -5, -11, 6, -1, 17, 5, 14, -1, 0, -4, -12, 5, 1, -6, 8, -4, -15, -21, 13, -4, -7, 3, 13, 11, -14, 3, 0, 2, 5, 0, 0}
, {-18, -3, 0, 6, 12, -13, 12, 16, 14, 3, 5, 2, 10, -2, -6, 11, -3, 12, -15, 8, -10, 13, -16, 8, 0, 10, 15, -19, -5, -15, -7, 5, 10, 12, -10, 1, 6, 14, 11, -2, -13, 2, 16, 6, -4, 13, -5, 12, -15, -5, 4, -8, 0, 4, 5, -13, -16, 13, -8, 6, -11, -5, -5, 14}
, {4, -16, 12, -3, 10, -15, -8, -15, 17, 11, -11, -10, -10, -8, -14, 14, -11, -18, -7, 9, -3, -5, 0, 6, -7, 6, -4, 12, -7, 8, 11, -13, 6, -9, 10, -9, -3, 14, -4, -2, -2, 8, -2, 6, 3, -8, -8, -11, -10, -10, -16, -3, -13, 5, -16, 3, 4, -15, -16, -8, 7, -5, 8, -11}
}
, {{-15, -15, -9, -11, 6, -11, -11, 3, 15, -14, 1, -9, 4, 1, -11, -10, -3, -1, 4, 5, 1, 6, 3, 0, -1, 5, 3, 9, -6, -9, -10, 0, -9, 14, -11, -12, -8, 14, 9, 1, -3, 1, -5, 7, -3, 13, -3, 12, 5, 3, -3, 12, 9, 1, 9, -14, 2, -6, -16, 16, -3, 4, -14, -14}
, {4, 1, 9, -3, 1, -15, 7, 10, 18, 4, 13, 13, 19, -10, 5, 7, -4, 7, 10, -4, 18, 13, 13, 2, 13, -5, -8, 9, 7, -7, 14, 12, -11, -9, -4, 1, -16, -7, -8, -3, 4, -12, -7, -11, 17, -7, 3, 1, 0, 10, -9, -1, -11, -11, -1, -6, 3, 14, 6, -15, -11, 11, 0, -10}
, {0, 10, 20, 8, 12, 4, -1, 0, 9, 5, -10, 16, -3, -9, -11, 19, -15, -13, 7, 3, 12, 12, 1, 7, -4, 14, 0, 12, 4, -6, 16, -10, 9, -13, 2, -4, -5, 4, 9, 17, 1, -14, 4, -11, -7, 5, -5, -9, -15, -14, -6, 4, 10, -4, 16, 6, -15, 2, -16, -7, 17, 6, -6, -10}
}
, {{1, 12, 5, -9, 3, 2, 14, 9, -12, 7, 12, 10, 14, 5, 16, 4, -11, 5, 12, 12, 11, -1, -4, -10, -1, 1, -4, 12, -12, -1, -16, 4, -2, -12, -13, 4, -4, -12, -3, -3, 8, 8, 1, -1, 0, -13, -9, 9, 8, 1, -1, 8, 0, 11, -12, -15, 9, 13, 11, 1, 10, -15, -10, -14}
, {3, -10, -6, 15, -1, 7, 9, 11, 9, 3, -16, -16, -5, -15, -13, 3, 11, 3, 7, -1, -11, -1, 0, 5, 5, -2, 10, -12, -16, 9, -7, -10, 9, -11, 6, 10, -16, -6, 2, -3, -15, 8, -13, 13, -8, 11, -13, -7, -3, 4, -2, -7, 12, -16, -10, -6, -2, -2, 5, -9, 12, 11, -15, -17}
, {-6, -3, -8, -1, -17, 11, -5, -18, 3, -10, -1, 7, -10, -8, 11, 6, -15, -4, 10, 11, 12, 1, -11, -1, -15, 11, -15, 12, 3, -14, 13, -12, -12, 5, -9, -5, 8, 6, -11, -10, -5, -15, -15, -17, -1, -8, -8, 14, 0, -14, 0, 12, 6, -3, -2, -1, -3, -7, -8, -7, 0, 8, -10, -4}
}
, {{-12, -6, 9, 20, -11, 1, -18, 5, -14, -12, -9, -7, -13, 16, 16, -2, -15, 1, 8, 9, 0, 11, -4, -10, 3, -1, 11, 2, 2, -16, -14, -17, 7, -4, 16, 12, -9, -17, 3, 9, -3, -7, -3, -2, 3, 9, -9, 8, -4, 5, 12, 6, -7, -5, 10, 0, 11, 4, -5, -4, 12, -1, 15, -10}
, {-1, 4, -12, 19, 3, 5, -10, -9, -14, -10, -10, 12, -4, 4, 12, -9, -9, -12, -5, -15, -12, 0, 9, 13, 4, 1, 10, 1, 10, 2, 2, -8, 2, -11, 5, 4, 14, -11, 5, 0, -1, -12, -11, -13, 8, -7, 3, -16, -9, -13, 13, -3, 16, -15, 15, -2, -3, -15, 1, -8, 13, -7, -6, -15}
, {2, 2, 3, 17, -12, 14, -15, 12, -4, 10, 5, -5, 1, -14, 17, -4, -9, 9, -5, 5, 6, -3, -12, -5, -16, 2, 9, 6, -8, -1, -8, -13, -10, -7, -3, 7, -6, -8, -3, 0, 3, 4, -16, 8, -8, 0, -9, 14, -10, 4, -10, -1, -6, 4, -16, 2, -2, -9, 12, 3, 9, -7, -9, -7}
}
, {{6, 4, -6, 6, -13, 13, -6, -6, -8, -16, 3, -17, -12, -13, -17, 11, -10, -11, -11, -12, -9, 10, -4, 11, -13, -11, 3, 5, 8, -12, 13, 5, 3, 8, 12, 3, -12, -8, 15, -5, 11, 11, -6, -10, -4, 6, -2, -8, 0, 3, -7, -1, -5, 6, 2, -5, 1, -13, -14, -7, 10, -4, 6, -17}
, {-6, -11, -7, -8, -1, -3, -13, 1, 10, -8, 4, -9, 12, -3, 5, -1, -4, 3, 1, -2, -2, 5, 3, 8, -13, 13, 15, -7, -5, -7, -9, 0, -5, 10, 8, -17, -17, -7, -1, 0, -8, 14, -15, 6, 16, -2, 0, -6, 4, -8, 5, -9, -16, -15, 14, -7, 11, -7, 9, 0, -2, -12, 11, 10}
, {0, -6, 2, 1, -7, -8, -6, -9, 8, -9, -14, 8, 13, -1, 3, -9, -3, -2, -12, 5, 2, 2, 10, 7, 14, 2, 1, 0, -7, -2, -4, -14, -11, -8, -8, -9, -7, 11, 15, 5, 13, -17, 8, -15, -6, 8, -4, 11, 1, -15, -13, -14, 12, 2, -14, 1, 13, 11, -16, -8, 13, -9, -4, 7}
}
, {{-2, 1, 5, 3, -17, 7, -12, 7, -5, -9, -11, 5, -1, -13, -5, 10, -7, -6, 2, 15, -14, -17, 14, -15, -4, -8, -9, 6, -14, -16, 3, 3, 1, -5, 1, -19, 1, -13, 10, 0, -8, -9, -18, 1, 12, -13, -8, 6, 0, -14, 6, -8, -11, -6, -8, 4, -3, -9, -18, -18, -6, 4, -16, 8}
, {-2, 4, 12, -1, 1, -7, 6, -5, 6, -2, -14, -14, -4, -20, 1, 5, -6, -16, -8, -7, 9, -4, -8, -14, 0, -4, -12, -8, 5, -3, 9, -17, 4, 2, 1, 10, 3, -11, -1, -5, 8, 9, 4, 2, -2, -2, 6, -3, 1, -8, 7, -5, -1, 10, 14, 1, -11, -8, 6, -6, 6, 6, -13, -10}
, {9, 8, -1, 7, 9, -13, 0, 6, 9, -10, 9, 3, 2, -12, -8, -7, 0, 4, 4, -16, -15, -7, 1, -12, -10, 15, 9, 1, 15, -18, -13, 2, -12, -3, -6, -16, -1, -11, -9, 17, 7, -10, -3, -16, -13, -13, -9, 8, 5, 12, 6, 6, -14, 14, 14, -16, -8, 0, -13, 9, -11, -6, -13, -20}
}
, {{1, -6, 3, 9, 13, 0, 1, 13, -7, -9, -12, 5, -8, 15, 15, 0, 17, -3, -8, -12, -17, 11, -1, -7, 1, -7, -8, 0, 1, -5, -5, 3, 9, -8, -16, -3, -12, 13, 9, -10, 7, 8, 2, -16, -4, -9, 5, -7, -3, -3, 5, 6, 7, -19, -6, -6, -13, -14, -14, -8, 15, 13, -3, 2}
, {8, 0, 3, 1, 1, -7, 2, -5, 18, -6, -9, -5, 16, -16, -14, 18, 10, -12, 7, -14, -16, -12, -5, -13, -4, 11, 12, -11, -15, -9, 4, -8, 11, -1, -8, -7, 15, 13, 9, -6, -1, -15, 3, 3, -13, 4, -5, 6, -6, 9, -2, -11, -17, -15, 1, 14, -17, 14, -10, 13, 10, -15, 0, -11}
, {5, -5, -8, -1, 5, 5, -4, -8, 4, 10, 11, 3, 13, -4, -11, -12, -1, -13, -2, -14, 11, 15, -7, -2, 8, -6, -11, 2, -17, -1, 18, 0, -5, 1, -8, -1, 13, 11, 3, -12, 3, 6, -3, -16, 7, -4, -2, 9, -10, -8, -13, -4, -11, -12, 5, -5, 9, -6, -10, -1, 1, 10, 14, 14}
}
, {{-18, 11, -5, -8, -16, 3, -8, 0, -12, -4, 7, -13, 14, -11, -6, 2, -8, -5, -2, 3, -4, -12, 5, 8, -1, 8, -11, -15, -14, 14, 16, -4, 6, 16, -1, 5, -3, 8, -19, 8, 4, -16, -11, 19, -2, 4, -7, -10, -4, 11, -1, 5, 10, -10, -10, 5, -6, -15, -16, -3, 3, -10, -14, 13}
, {5, 11, -6, 3, -5, 5, -1, -4, 6, -13, 1, -1, 15, 11, -9, 12, 10, 3, 7, 8, -8, 12, 1, 3, 5, 15, 9, -1, 9, -10, 14, 7, -3, -2, -17, 16, 16, 18, -11, 3, 4, -2, 5, -7, 2, 3, 17, -12, 1, 1, 1, -3, 3, -7, 14, 15, -8, 2, 1, 10, 14, 7, 0, -12}
, {10, -7, -9, -5, -14, 14, -13, -14, 4, -7, 7, -3, 12, -2, 3, 4, -15, -10, -6, 3, 1, -11, 9, -9, 15, 4, 11, 12, 1, -8, 5, 12, 3, 14, 10, -5, -6, -8, -10, -10, -5, -6, -1, 0, 4, -13, -11, 3, -16, -12, -7, 17, 10, -2, -8, -1, 4, -1, 7, -15, 0, -11, -15, -5}
}
, {{0, 10, 17, -17, 11, 12, -4, 1, 1, -13, -4, 3, 7, -16, -18, -7, -13, 8, 16, 5, -10, -16, 10, 6, 9, -14, -3, -2, -14, -16, -3, 7, -10, -8, -14, 2, -4, -8, -9, 14, -4, 4, -10, -19, 12, -3, -6, 8, 1, 5, -6, -12, 1, 8, -2, -6, 1, 12, 14, 3, 3, -14, -14, 17}
, {17, -12, -3, -9, -14, -18, 8, 16, -2, -11, -12, 13, -2, -3, -14, 8, 6, -1, -3, -5, -13, 8, 12, -14, -3, 1, 8, 5, -15, 13, -8, 5, -2, 11, 11, 4, 6, -2, -7, -9, -9, 6, 2, -19, 8, 11, 5, -17, -3, -11, -4, 3, 4, 0, 11, -13, -16, 10, -7, 0, 14, -4, -14, -11}
, {13, -11, -1, 0, -12, -11, 15, -7, -11, 11, 8, -8, -12, -1, -18, -4, -11, -6, 11, 13, -3, 10, -5, -4, 1, 9, -12, 7, -10, 5, 6, 8, -4, 5, 5, -7, 0, 20, -9, 14, 23, -10, 13, 11, 11, -9, -14, -5, -5, 13, 4, -16, -9, -15, 16, -13, -19, 12, -1, 8, 11, -3, -11, -14}
}
, {{1, -12, -14, 4, 0, 7, 15, -12, -15, 3, -18, 3, -3, -6, -4, 7, 18, 23, 9, 4, -2, -2, -4, 13, -8, -7, 0, 5, -9, -2, -4, 4, 12, -3, 4, 10, -16, -17, 1, 2, -15, 8, 8, -18, -12, 4, -12, 6, -5, -9, 14, -14, 5, -6, 1, 11, 4, -9, 10, 0, 4, 0, 12, 12}
, {10, -9, -15, 10, 10, 12, -9, 10, -1, 6, 2, 10, 0, 7, 14, 2, 0, 13, -15, -6, -7, 12, 8, 1, -14, 10, 4, -15, -10, -1, -20, 3, 11, -13, 8, -6, 7, 10, -5, -12, -13, 16, -15, 10, 3, 6, 9, -6, -17, -8, 9, -12, -7, -15, -7, -1, 0, -6, -14, 9, -12, -3, 2, -18}
, {-1, -13, 4, -9, -8, -17, 6, -8, -14, -18, 2, 4, -6, -10, 4, -11, 7, 9, 2, -16, -3, -6, -1, 11, -11, -17, 0, -10, -4, -9, -10, -7, -10, 9, -7, 7, -18, -15, 8, 1, 11, 6, -14, -9, -10, 13, -18, 16, -3, -19, 11, -16, 2, 6, -19, -5, 7, 14, 11, 1, 5, 0, 8, 2}
}
, {{9, 4, -9, -6, 3, -16, -6, -12, -7, 11, -10, 4, 2, 4, 3, -1, -3, -9, 1, 3, 6, -14, -2, 3, -19, -6, -8, 10, -14, -11, 0, -3, -14, -6, 15, -17, -10, -17, 7, 7, -4, -5, -4, -1, -6, -6, -6, -3, -7, 0, 1, 10, -8, -7, -12, 11, -2, -12, 9, -16, 1, 12, -4, -11}
, {9, -9, -6, -12, -15, 11, 5, 11, -8, 8, -14, 9, 12, -12, -4, -8, 2, -6, -1, 10, -12, 13, 10, -8, -12, -11, 6, 1, 5, 5, -11, -8, -12, -20, 1, 1, 6, -5, -8, 6, 7, -6, -14, 15, 16, 11, -19, 10, 14, -9, 6, -4, 15, -4, 7, -1, 4, 11, 14, -4, -13, 9, -10, 1}
, {-14, -2, -3, -13, -1, -3, -20, -20, -14, -1, -2, -16, 14, -17, -1, 9, -10, -3, -16, -2, -1, 13, 12, 4, 12, 13, 7, -11, 7, 9, -14, 11, -10, 2, -7, -18, 8, -16, -15, 6, -9, 10, -11, 8, 3, -15, 10, -12, 13, 2, 6, -12, -5, -1, -2, -8, 0, 5, -4, 12, 3, -12, -8, -6}
}
, {{-12, -13, 16, 3, -16, -2, 12, 1, -1, -2, -1, 6, -3, 11, 10, 14, 5, -3, 6, -2, 0, 10, 10, -5, -6, -13, -6, 7, 15, 12, -7, -13, -14, -8, 5, -11, 2, 7, 2, -13, -18, 16, -10, 2, 14, 14, 8, 5, -1, 1, 7, -17, 10, -3, -15, -10, 1, 4, -11, 7, 10, 1, -5, 0}
, {-10, 6, 0, 5, 11, 13, -14, 4, 8, -5, 9, -15, 4, 12, -6, -10, -6, -1, -5, -7, 1, 1, -13, -6, 7, -9, -9, 10, 0, -4, -14, -7, -1, -13, -13, 12, -2, -4, 0, -9, 4, -15, -14, 8, -16, -7, 12, -6, 4, -5, 3, 2, -5, 4, -11, 11, 14, -9, 9, -13, -9, -15, 1, -3}
, {-11, -16, -4, 13, 3, -5, 9, 1, -2, -14, 10, 6, 6, -16, -3, 0, -11, -7, 11, 0, 6, 14, 10, -11, 5, 12, -4, -5, -1, 14, -4, -13, -7, -5, 4, 2, -4, -9, -4, 11, -10, 7, -2, 6, 10, -11, -5, 14, 12, 9, -1, -12, -4, -15, -9, -9, -6, -11, -2, 3, -6, -14, -2, 8}
}
, {{-8, 15, -6, -6, 11, 9, -7, 17, -6, -13, -12, 5, 7, 11, 8, 0, 12, -8, -10, -15, 6, 11, 12, 3, 3, -13, 14, 5, 2, -8, -4, -2, 10, -5, -6, -4, -6, 14, -18, -9, -17, 5, 4, -10, -7, 11, 14, -15, 0, -6, -10, -5, -13, -11, 13, -2, -11, -4, -4, -2, 8, -9, -6, -10}
, {-7, -7, -14, 0, -6, -13, 8, 13, -9, 7, 4, -4, 11, 0, -8, 13, 8, -14, 4, -5, 7, 5, -3, -14, -2, -9, 2, 5, 10, 12, 9, 2, 11, 8, -15, 10, 5, -2, -17, -12, -10, -13, 7, -5, -2, 13, -2, 10, 6, 2, -2, 15, 12, -14, 5, 1, -1, 3, 11, 13, -17, 9, -16, -3}
, {3, -12, 9, -5, -15, -15, -8, 2, 3, 1, 0, -15, 6, -2, -1, 5, -8, 0, -4, 1, -2, 13, -12, -7, 5, 10, 9, 11, -10, -14, -1, -2, -6, -2, -4, -1, 2, 15, -18, 13, -16, 12, -1, 5, -7, 15, 12, 15, -10, -1, -1, 2, 13, 3, 8, 2, 5, -9, 11, -13, 2, 2, 8, -13}
}
, {{10, -13, -5, -12, 12, -1, -7, 11, 19, 1, 12, -1, -8, -8, 2, -8, 8, -12, -16, 5, 3, -14, -13, -6, -11, 2, 6, -11, -5, -6, 5, 1, 4, -12, -3, -19, 3, -7, -11, -11, 7, -5, -12, -6, -11, 5, -12, 6, -10, -9, 14, -4, -6, -15, -13, 12, 3, 6, 8, -4, 5, -16, -10, -9}
, {16, -8, -11, -1, -10, -2, -4, -3, 16, 12, -6, -18, 4, -2, -13, -12, 4, 0, 6, -4, -3, 5, -10, -16, 0, 1, 9, 1, -4, 13, 0, -8, -5, -3, -9, -4, 1, -1, 12, -13, -11, 4, 5, 15, 12, 6, -4, -11, 5, -4, 2, -2, 11, -1, -11, 11, 4, 10, -10, -9, -19, 7, 13, 5}
, {-1, -6, -6, -3, 6, -14, 3, 5, -1, 7, 9, -3, -6, -8, 1, 7, -1, 6, -9, 18, 7, -13, 14, -14, 1, -5, -7, -1, -7, 9, -10, 4, -9, -19, -13, -2, -13, 2, 10, -18, 1, -1, -3, 5, -9, 15, -15, 17, -2, 1, -10, 14, 3, -6, 0, 8, -16, -4, -18, -11, -5, -2, 13, 14}
}
, {{-4, -7, -11, 1, -9, -7, -4, -12, -13, -11, 2, -1, 4, -3, 9, 12, -2, 24, 11, 5, 7, -1, -11, 14, 15, 9, 6, 15, -11, 3, -11, -6, 1, 6, -9, 6, 3, 5, 17, -19, -5, -3, 1, -18, -15, 23, -1, 17, 5, 3, 12, 5, 9, -8, 11, 5, 14, -8, -5, -11, -15, -4, -2, -7}
, {6, -17, -14, 4, -9, 8, 15, -5, -13, -6, 12, 5, -12, -4, 21, -12, -5, 6, 7, -11, -19, -10, -6, 15, -7, -11, 0, -6, 2, -11, -6, 17, -4, 12, -10, -10, -7, -23, 4, -14, -5, -9, 14, 8, -14, 7, -5, 7, -9, 16, -6, 2, 12, -12, -13, -8, 2, -16, 2, -1, -5, 11, 3, -5}
, {20, 1, -7, 16, 4, -2, 0, -14, 6, 4, -7, 3, -12, 2, 4, -8, 14, 10, 4, 15, -12, 10, -13, 5, -12, -10, 2, 1, -14, 7, -9, 0, -8, 13, -5, 4, -11, -12, 2, -19, -11, 2, 5, 9, 15, 12, -3, 13, -11, 12, 1, 3, 10, 6, -8, -13, -16, 0, 2, -8, -13, -12, -4, -9}
}
, {{-12, -11, -12, 12, 5, -9, 13, 12, -13, 3, -2, -1, -12, -10, -7, -3, 11, -9, 1, -5, 9, -9, -3, -3, 5, -11, -3, 6, -14, 6, 3, 7, 5, -12, -10, -11, -8, -7, 3, -1, -4, -7, -1, -8, -13, -11, 9, -2, -14, -12, -14, 12, -12, 3, -18, -8, -18, 11, -8, 6, -11, 7, 6, -1}
, {-1, -17, -5, 2, -14, 2, -6, 6, 0, -10, -7, -2, -2, 10, 13, -12, 8, -1, -4, 4, -4, -3, 10, -4, 5, -13, 6, 3, -17, -14, 0, -2, -11, -5, -17, 2, 5, -2, 16, -13, 16, 6, -9, 7, 1, 11, 10, 8, -18, 5, 3, 11, -8, -9, 7, 7, -2, 12, -7, 13, -3, 16, -13, 14}
, {-3, -7, 4, 10, -10, 11, -6, -4, 11, 13, -12, 7, 11, 6, -13, -4, 2, -6, 6, -10, -11, -6, 6, 5, -8, -16, -4, -4, -12, 14, -2, 10, -10, 0, -2, -14, -11, 12, -9, 1, -11, -7, -10, -19, 0, 7, -15, 12, 10, 6, 4, 1, -13, 8, 2, -11, 10, 0, 6, 5, 1, -13, -1, -9}
}
, {{-4, 2, 0, 9, 14, -5, -17, -4, 12, 6, -9, 15, -1, -9, -16, 8, 10, -13, -13, -12, 3, -1, -11, -6, -2, 8, 5, 16, -6, 13, 13, 3, 13, 4, -3, 8, -15, 3, -7, -10, -11, -11, -1, 8, -12, -7, -11, 1, -5, 14, 9, 13, -5, -10, -4, -1, -6, -3, -14, 17, 2, 2, 10, 6}
, {-13, -7, 7, -2, -5, -17, -1, -12, -13, -8, 0, -3, -1, 2, -4, -6, 9, -14, -15, -2, 9, -3, -2, -3, -2, 2, 10, 2, 11, 3, 16, 17, 14, 5, 3, 0, -2, -3, 11, 15, -8, 9, 7, -18, 3, 9, 12, -17, -14, 3, -19, -4, 3, -17, 14, -14, 10, 21, -8, 10, 0, 6, -11, 9}
, {4, 20, 7, 5, 12, -15, 4, 15, -6, -15, 8, 6, -8, 8, -21, 18, 10, 8, -2, -13, -13, -10, -4, 12, -10, -10, 10, -8, 1, -3, -9, 0, 16, -15, -11, -3, -4, -14, -15, -12, 5, -14, 15, -11, -6, -2, -2, -3, 9, -12, -2, -7, -9, 9, -4, 10, 0, 16, 9, 0, 4, -8, -8, 5}
}
, {{14, -2, -16, 6, 9, 6, 13, 9, -7, 6, -6, 11, 12, -1, 4, -6, -3, -15, 0, 8, 4, -14, 6, 1, 7, 6, -2, 10, 13, -16, -16, -1, -7, 0, -5, 7, -7, -16, -9, 14, 7, -13, -8, -9, 9, 11, -6, -1, 16, -16, -12, 0, -8, 11, 9, 8, 9, -16, 11, -5, -15, 10, 3, -7}
, {-6, -1, 0, 11, 15, 7, 8, -5, -8, 3, 5, -1, 8, 10, 6, 4, -3, 1, 13, 10, 4, -9, -13, 12, 4, 11, 15, 15, 11, 13, 10, -9, -2, 4, 15, -10, -3, -10, 1, 9, 4, 4, -12, -5, -5, 1, -6, -22, 17, -14, -2, -1, -14, 16, 12, 13, 17, -14, -13, 0, 3, -3, -5, 6}
, {-6, -4, 3, -16, -10, 12, 1, 13, 2, 7, 13, 14, -8, -7, 12, 9, -15, 6, -16, 13, 14, 5, -18, -11, 2, -4, 7, 3, 9, -12, -12, 3, 7, -6, 3, -8, -1, 7, -5, -1, 8, -1, 15, -4, 0, -1, -5, 3, 5, -9, 11, -2, -6, 1, 4, 6, 4, 4, -7, 2, 12, 1, -17, -6}
}
, {{-4, -3, 14, 12, 1, 7, 3, 12, -11, 5, -5, -8, -5, 2, -12, 8, -2, -10, 2, -17, -7, 0, 7, 2, 1, -7, -6, -16, -5, -13, -6, 6, -7, -9, 12, -8, -8, 1, 4, -9, -11, -2, 6, 1, -7, -3, 9, 3, 12, -4, 10, -12, -13, -9, -6, -12, -16, 6, -8, 6, -6, -13, 0, -12}
, {-11, 15, -7, 6, -6, -6, 14, -7, 6, -12, -9, -4, 15, 9, -20, 7, 16, 11, 10, 0, 10, 8, 5, -7, 5, -15, 6, -4, -6, -12, -2, 1, -8, 0, -4, 1, -4, 0, 2, -3, -4, -12, 10, -9, -6, 11, -8, 2, -9, -6, 2, 7, 7, -14, 3, 10, -4, 2, -8, 3, -11, -5, 1, 2}
, {1, -14, 3, -8, 1, -6, -3, 15, 12, -8, 6, 4, 13, -9, -18, 10, -6, 9, 4, -14, -7, -11, -3, -13, 2, -16, 0, -8, -5, -16, 18, 5, -1, 8, -11, -4, -7, 13, -2, -9, 2, 6, -10, -4, 4, 4, 7, -14, 7, 4, -13, -6, 10, -13, -5, -1, 9, -10, 11, 13, 3, 9, 3, -16}
}
, {{-17, -6, 13, 6, -10, 6, -15, 0, 4, 1, 6, 3, -3, 2, 12, 19, 4, -17, -16, 2, -12, -3, 16, 11, -13, -10, 2, -7, -4, 4, -6, 2, -7, 2, -11, -3, -8, 20, 14, 15, 11, -9, -14, 8, 5, 7, 13, 6, 7, -14, -6, -11, 8, 9, 17, -5, 3, 3, -5, -7, 11, 8, -8, 9}
, {-18, 2, 11, 2, -1, -13, 6, 17, 13, 17, 6, 0, 13, -14, -11, -1, -1, 11, 1, -2, -12, 1, -8, -2, -2, 7, -8, -1, -15, 3, 4, 12, 16, 4, -8, -11, 0, -10, -8, -1, -1, 1, 13, -6, -5, 11, -14, 0, -1, -12, -10, 0, 3, -14, -8, -10, -8, 16, -7, -8, 4, -3, -4, 0}
, {11, 0, 8, -12, -9, -8, 7, -5, -2, -8, 7, 13, 7, -1, -18, -1, -3, 7, 4, -6, -7, 3, 4, 8, -13, 9, 16, 10, 6, -1, 6, 5, 10, 4, 3, -14, -8, 4, -8, 1, -8, 2, -1, 8, 4, -20, -6, 10, 13, 11, -5, -7, -21, 1, -5, 0, 4, 4, 11, 8, 7, -3, 7, -13}
}
, {{-17, -15, 9, -5, -8, -6, 12, -6, 7, -11, -11, 12, -8, 14, 11, -8, -4, -10, -7, -8, 5, -4, -5, 2, 12, -4, 3, 3, -6, -16, 7, -4, 15, -6, -14, 7, 12, 8, -10, -14, -20, -15, 10, 15, -14, -15, 13, -2, 5, -4, -9, 2, -9, -5, -3, 7, -13, 4, -7, -12, -10, -7, 8, 2}
, {-1, 14, -2, -7, 1, 5, 2, -15, -11, 12, -2, -2, -6, -3, 3, 4, 12, 8, 8, 6, -1, 8, 1, 12, 0, -1, -6, 10, 8, -10, -10, 7, 1, -4, 5, -4, -15, 11, -3, -9, -5, -9, -12, -14, 6, 0, 1, 13, -9, -15, 1, 3, -6, -14, 8, 15, -4, -13, 9, 1, -17, 9, 12, 14}
, {-13, 5, 1, -9, -1, -16, 5, -11, -9, 6, -6, 12, -9, 11, 11, -15, 9, -16, 11, -9, 4, 7, -10, -1, -5, -10, -6, -1, -3, -3, -12, -9, 4, 7, -19, -6, 3, 4, -21, -11, -10, -5, -14, 15, 5, -3, 8, -11, -7, 1, -7, 3, -12, -8, 12, -14, -10, -14, 7, -11, -17, -9, -7, -18}
}
, {{-9, -3, -3, 2, -16, 10, 9, -7, 11, -1, 10, -17, 4, 13, 11, 14, 6, -13, 16, 14, -8, 15, -9, -15, 17, 0, -14, 10, 5, 7, -2, -7, -2, 1, 14, 9, -16, -13, -14, -8, -7, -2, 15, 13, 14, -5, -2, 6, -8, -13, -3, -12, -18, -1, -10, -14, 6, -15, -15, -4, 9, 3, -1, -13}
, {7, -11, -16, 2, 0, -17, -4, -13, 13, -15, 1, -4, -12, 8, -2, 12, -10, 8, -7, 15, -2, 7, -6, -7, 4, -1, -11, -13, -5, 12, 14, 7, 15, -10, -5, 1, -10, 12, 10, 13, 2, 11, -18, 1, 6, 19, 11, 8, -3, 0, 2, 2, 0, -8, -13, 3, 8, 13, -10, 8, -7, 3, 15, 7}
, {-12, 10, 9, 4, 9, -16, -8, 5, 5, 7, -3, -7, 11, -14, -15, 6, 15, 4, -1, 10, 16, -10, 11, -6, 1, -2, -5, 11, -8, 12, -4, 3, 13, 16, 1, -7, -16, 16, 3, 12, -16, 2, 2, -5, -12, -9, 11, -5, -9, 10, -3, 7, 12, 13, 5, -2, -13, 13, 15, -16, 10, -3, -15, -14}
}
, {{-4, -13, -6, -6, 2, -6, -1, 9, 8, 3, 3, -12, -10, 2, -3, 13, -2, -14, -8, 8, -8, -15, -17, 8, 1, -10, 4, -9, 9, 4, 13, 7, -13, -18, -8, -16, -12, 13, -12, -9, 7, -10, 5, 14, 6, -4, -9, -6, -3, 6, 7, -4, -14, -17, 5, 15, 14, 11, -12, -15, -17, 0, -3, -1}
, {9, 5, 5, -1, 11, -9, 5, -13, -9, -12, 12, -11, 5, 1, 9, 9, 7, -16, 5, -16, -14, -7, 2, -1, -3, 2, -1, 11, 7, -1, -6, -10, 16, -11, 10, 8, 4, -17, 7, -9, 2, -16, 6, 14, 10, 13, -12, 6, -1, -15, 0, 8, -10, 2, -11, 7, -10, 6, 5, -14, 14, 1, -16, 12}
, {-12, -4, 16, 8, 7, -1, -12, 15, -10, -2, 15, 8, 6, -8, -13, -7, 2, 4, 13, 6, -5, 9, -10, 14, -16, 1, 8, 11, -17, -8, -10, 14, -3, 0, -11, 6, -9, -14, 9, -15, -6, -2, 0, -9, -7, 4, 5, -2, -4, -8, -2, 13, -16, 8, 7, 20, 0, -9, -17, -11, -11, 0, -15, -9}
}
, {{13, -5, 11, 2, 5, 6, -16, 10, 15, 12, -7, -3, 0, 2, 1, -12, -1, 0, 8, 8, 8, -7, 9, -18, -6, -9, 14, 2, 7, -14, -9, 2, -21, -9, -11, 9, -19, 12, 2, -10, -4, -10, -10, -3, 5, 0, -8, -12, -11, -15, 2, -18, -1, 5, -12, 11, 3, -5, 12, 9, -11, -8, -3, 9}
, {2, 2, 13, -15, -9, -11, 2, 10, -15, -4, 2, 12, -15, -13, -8, -4, 8, -10, -17, 5, 9, -9, 8, -19, -1, 4, 16, 9, -4, -10, 4, -6, -11, 13, -14, -1, 9, -9, -18, 9, -14, 2, 12, 10, -8, -4, -1, -9, -12, -14, 2, 3, 12, -10, -11, 0, 13, -14, -7, -15, 13, -3, -3, -15}
, {-8, 12, 1, 3, 11, 13, 3, -20, 12, 9, -6, 14, 7, 14, 1, -10, 7, 15, -1, 10, 2, 9, -14, -3, 6, 11, -3, 7, -3, -12, 0, -15, -13, -10, 9, 7, -19, 10, -1, 3, -9, 2, -9, 6, 11, -1, -15, 13, -6, -3, 12, 9, 7, -8, 12, -7, -1, -11, -8, 14, 8, 1, -11, 8}
}
, {{-6, 8, 17, 4, 14, 15, -1, -8, -5, -12, 5, -3, 1, -7, -13, 6, -10, 10, 0, -10, -8, 6, -11, 14, 3, 3, 6, -7, 7, -10, 11, -5, -7, -3, 12, 11, -6, -4, -6, -14, -3, -12, -4, 7, 3, 2, 13, -13, 8, 4, 6, 7, -12, 8, 0, -13, 7, -8, -12, 8, -8, 15, -11, 6}
, {1, -16, -7, -2, -2, -6, 13, -1, -3, 6, -7, 1, -4, 3, -4, 14, -3, 17, -11, 3, 5, 12, 7, -5, -8, 6, 10, -3, -1, 12, -2, 2, -1, 12, -11, -5, 8, -7, 5, 12, 11, 1, -16, -9, -13, 0, 3, 10, -2, -7, -3, -6, -11, 1, -8, -13, -8, 11, -16, -8, -10, -15, -4, -13}
, {2, 9, -10, 14, 7, -8, 4, -8, 3, 3, -15, -2, -13, 5, -16, -5, -11, 3, 8, -12, 12, -17, 12, -6, 6, 4, 3, 7, -16, -18, 16, 5, -9, 9, -6, -4, 9, 11, -13, -9, -2, -4, 6, -13, -15, 4, -13, -1, -7, -12, 3, 2, -11, -11, 5, -11, 8, 15, 11, -6, -12, -3, 9, -14}
}
, {{8, 3, -3, 9, 7, -6, 3, -12, -10, 10, -14, 0, 15, -5, 11, 9, 5, 6, -5, -16, 3, -5, 9, -13, 11, 14, -6, -17, -5, -2, -3, -10, 8, -12, 15, -3, -6, 8, 7, -1, 6, 1, 16, 8, -6, 4, 0, 11, 11, 11, -16, -15, -6, -14, -13, 10, -5, -6, -12, 14, -6, 4, -9, 1}
, {-13, 10, 5, 5, 12, 4, 7, -5, 3, -7, 14, -2, -6, 10, 4, 13, -4, 4, -7, -8, 4, 0, 6, 6, -10, -10, -3, -19, 3, 16, 17, -12, -11, -10, 9, -12, -5, -2, -15, -12, -14, -3, -3, -10, -14, -6, -7, -3, -19, -3, 6, -3, 5, -3, -1, 9, -4, 9, -9, 3, -13, 3, -2, 16}
, {7, 7, -2, 2, -16, -2, -8, 12, 4, 15, -16, 13, -2, -8, -15, 15, 11, 0, -8, -17, -13, -9, 7, -11, 7, -2, -5, 11, -7, -2, 2, -5, -10, 1, 3, 9, 5, 4, 3, -7, -11, 13, 7, -15, -5, 14, 3, 10, 0, 9, -10, -10, -8, 9, -12, 9, 4, -3, -3, -1, 14, -1, 11, 3}
}
, {{-12, -7, 3, 11, 4, -7, -2, -4, 15, -16, -3, 3, -7, 12, 8, -3, -16, -1, -4, 7, -10, -10, 11, -13, 13, 10, -13, 9, -1, -16, 14, 3, -12, -12, 3, -10, -17, 8, 6, 15, 14, -5, 3, 5, 9, 3, -2, 4, -7, 7, -4, 4, 9, 16, 3, -11, -1, 3, 9, 9, 8, 10, -15, 9}
, {15, 13, 0, -9, -12, -8, 14, 4, -3, 2, -3, 10, -9, -18, -13, 13, -3, 0, 0, 7, -7, -17, 4, -9, 10, -8, -12, 4, -12, 1, 4, -4, -13, -11, -4, -7, 0, 8, -6, 10, 9, 0, -2, -11, 13, 2, 1, 5, -6, 11, -11, 11, 0, 9, -18, -13, -6, 17, 6, -17, -7, 8, 6, -9}
, {-3, -11, -7, -3, 11, -10, -6, 4, -2, -12, 6, 8, 7, 2, 8, 5, -6, -7, 5, -12, 13, 9, -3, -9, 7, -5, 12, -11, -2, 9, -10, 14, 12, -18, -14, 6, 11, 10, 2, -2, -1, -6, -14, -15, 12, -10, -15, 8, 18, 4, -11, 14, -9, 7, 7, -2, -6, -1, 8, -4, 13, -16, -18, 10}
}
, {{-6, 14, 15, -11, 1, -16, -3, -1, 13, 9, -8, 15, 9, 2, -12, -7, 5, -13, 1, -8, 1, 6, -11, -3, 0, -11, 1, 12, 1, 2, -9, -11, 9, 2, -14, -12, 10, 13, -9, -3, -6, -9, 11, -1, -11, 8, -1, 12, -9, -6, -12, 9, -4, -1, 14, -6, 1, 17, -4, 1, -7, 6, 2, -5}
, {-12, -8, 0, 6, 12, 6, -4, -2, 15, 12, -14, -3, -12, 6, 0, 18, -8, 7, 5, -4, -13, -12, -8, -10, 12, -9, -1, 4, 10, -1, 13, -9, 6, -8, 5, 8, 3, 4, -4, 3, 6, 8, 2, -8, 5, 0, 2, -12, 7, -14, -3, 4, -10, -20, -4, 14, -14, 0, 12, -10, -4, -15, 9, 8}
, {-9, 12, 1, -2, 7, 11, -11, 0, 8, -3, 8, -4, 14, -13, -16, 18, 9, -10, -12, 16, -9, 13, -13, 3, 7, 10, 10, -4, -13, -4, 12, 15, 7, -11, 11, -13, -13, -12, -2, 15, 6, -4, 6, -15, 9, 15, -15, -3, -12, 10, 4, -7, -13, 9, 11, -13, 13, 7, 9, 6, -14, -14, 8, -14}
}
, {{-2, -8, -5, -7, 0, 8, -6, 3, -4, -6, 8, -7, 11, 0, 12, -7, 5, 2, -2, 6, -14, 7, -2, -18, -8, 7, 5, -3, -11, -7, -3, 0, -1, 8, 6, -2, 9, 15, 12, 11, 0, 11, -1, 0, -11, 17, 11, -14, -15, -16, -8, -16, 4, 0, -5, 10, -1, 15, -17, -2, 2, -1, -12, -8}
, {0, -7, -5, 17, -15, -4, -9, -5, 3, 6, -17, 3, 0, 13, -17, 6, -11, 0, 10, 5, -6, -15, 12, -11, 9, -16, -6, 8, -7, -5, 9, -12, -15, -6, 13, -4, 9, -3, 4, 12, 7, -3, 5, -18, -9, -3, 14, 8, 8, -8, 6, -20, 4, 11, -5, -8, -1, 1, 0, -7, 2, -7, -16, 12}
, {7, -3, -9, 8, -4, -5, -5, 6, -11, 8, -13, -13, 0, -9, -10, 2, 2, -7, -7, -8, 5, -3, -8, -11, 15, -16, 2, -6, -13, -1, 16, -11, 11, -13, -2, -14, 13, 10, 10, 12, 15, -2, -9, -4, 1, -9, 15, -9, 12, 0, -14, -2, -3, 3, 14, -8, -9, 2, -13, -12, 3, -4, -17, 6}
}
, {{-14, -9, -11, 2, 4, 13, 13, -3, -8, -6, 11, -17, -10, -1, 1, 12, -5, -2, 14, -6, 10, -2, -14, -2, -19, 7, 4, -14, -15, -2, -6, -13, -4, 12, 5, 11, -7, -11, 15, -3, -10, 11, -13, 9, -2, 9, -11, 1, 15, -5, 14, -3, 8, 3, 4, -12, -3, -14, -17, -12, -9, -1, 4, -4}
, {-1, -10, -10, -18, 7, 10, 8, 6, -9, -1, 13, -14, 14, -5, 13, 3, -7, -14, -14, -8, -9, 3, -6, -6, -4, 11, -1, 6, 10, -8, 0, 11, -3, -11, -6, -1, -3, 2, 15, -3, 7, -4, -9, -13, 18, -10, 6, 7, 16, -1, 9, -9, 12, 8, -2, 7, 9, -3, -11, -6, 8, 12, -3, 6}
, {18, 10, 9, 3, -13, 6, -9, 4, -15, -1, 13, 6, -12, -4, -1, 1, -17, -5, -10, 17, -9, -4, -15, -6, -15, 16, 12, -4, -7, 1, 5, -14, 5, -10, -15, -11, -1, -1, 12, 10, -7, -3, -1, 7, 11, 5, 5, -19, 1, 13, 11, 3, -15, 1, -7, 12, -10, -7, 4, 7, 9, -9, -8, -9}
}
, {{-14, 6, 5, 3, -16, 12, -13, -18, 6, 3, -3, 9, -6, 7, 9, -13, -19, -12, -15, 5, 1, -10, 7, 8, -13, -14, 16, -6, 13, -6, -1, -2, 0, -3, -11, -12, -15, 1, 11, -5, -9, -17, -2, 14, 3, -7, -8, 0, 13, 7, 18, -15, -1, 1, -12, -7, -6, -14, 13, 13, 12, -9, 4, -2}
, {-11, 11, -6, -12, -8, 12, 13, 1, -11, -18, -15, 7, 11, -13, -16, -1, -1, -2, 5, -7, -15, -13, -10, 2, -6, 10, -11, 10, -12, -11, 9, -10, 0, 7, -3, -2, 1, -11, 2, 5, 13, 12, 3, 15, 1, 12, 14, 2, 8, -1, -2, 3, 2, -5, -6, -14, 14, 9, 3, -9, -8, -11, -9, -15}
, {-10, 10, 17, -8, 2, -12, 14, -3, 9, -7, 15, 15, 8, 6, 8, 13, 14, -5, -14, -7, 0, 7, -12, -8, -1, -10, -10, -15, 9, -5, -2, 2, 4, -3, 17, 13, -6, 3, -16, -12, -5, -15, -9, 7, -11, -14, 14, 4, -2, -6, -9, 9, 0, 6, 16, -13, 3, -9, -14, 2, -1, -2, 8, 3}
}
, {{-13, 3, -18, 2, -5, -15, 11, -16, -17, -2, 3, 9, 8, 4, 12, -5, 22, 19, 7, 8, -15, 3, 13, 13, 11, 13, 3, -1, 7, 1, 11, -10, 13, -5, 14, -12, -13, -17, 5, -11, -8, -4, -16, -11, 12, 5, 8, 4, 8, -2, -6, 8, 8, -2, 14, 1, 5, -3, 4, -14, 14, 11, -7, -15}
, {-13, 11, -16, -11, 10, -6, -4, -5, -11, -10, 3, -5, -14, 16, -9, -5, -5, -8, -4, 7, -17, -16, 14, 6, -14, -2, -12, -17, -14, -13, -14, 6, 15, -17, -12, -10, -5, 7, -5, 5, 6, 1, 5, 11, 13, -8, 7, 26, 7, -12, -14, -6, 2, -2, -10, 4, 0, -15, -7, -14, 10, 4, -6, -1}
, {1, -10, 8, 2, 1, 0, 7, -11, 0, -2, 7, 2, -1, 9, -3, -8, 17, -9, -5, 0, -1, 15, -7, 6, -12, -3, -11, 4, -14, -15, -17, -3, -5, 12, 1, -14, -11, 10, -6, -12, 3, 1, 2, 8, -4, 8, 12, 15, -18, -2, 3, 1, 6, -14, -16, -1, -6, 0, 3, 5, 13, 2, 8, -15}
}
, {{-8, -14, -16, 8, -1, -8, -8, -17, -7, -1, -12, 11, 13, -10, 6, 5, -4, -16, -14, -5, -2, -9, 2, -7, 7, 12, 4, 13, -13, -3, -10, -12, -17, -3, -14, -1, 3, 2, -17, 9, 13, -2, 8, 9, 12, 0, -4, -13, -9, 4, 2, -10, 8, -9, 10, 5, -10, 2, 7, -2, -9, 3, 13, -15}
, {-11, 14, 14, -10, 0, 7, 11, -2, -3, -16, -5, -17, -13, 11, -7, -9, -7, 3, 0, -5, 13, -10, -16, -5, -2, -5, 0, 17, 2, 15, 0, -16, 8, 3, 7, -14, -17, -13, -13, -7, -6, 13, -1, -2, -11, 0, 6, -3, 1, -16, 9, -14, -4, -6, 9, 10, -14, 11, -4, -17, 6, 10, 5, -7}
, {-1, -6, -6, -13, -18, 0, -3, 10, 1, 6, -11, 6, -7, 12, 8, -11, 6, 1, -2, -1, 9, 12, -4, 14, -8, 11, 8, 18, 0, 6, 9, 16, 1, -4, 12, -3, -14, 14, -12, -11, 11, -10, -5, -11, 9, 1, 2, -13, -12, -14, -2, 9, 13, 14, -11, -3, 8, -14, 5, -1, 5, -12, -1, -6}
}
, {{9, 5, 7, -10, -1, 2, -5, -1, -19, -7, -8, 7, 4, -16, 16, 9, -14, -16, -8, 12, -12, -11, -7, 14, 2, 4, -10, 9, -11, 9, -16, -9, 11, -9, -15, 7, -14, -7, 3, 8, -3, 11, -11, 11, -12, 5, 7, 7, 6, -1, 18, -5, 14, -7, 6, 4, -7, -6, -3, -5, -5, -1, -8, -9}
, {-1, 11, -6, 1, -5, -14, -16, -2, 4, 8, -8, -15, 14, -3, -4, 11, 11, 6, 6, 13, -7, -8, 8, -8, 2, -10, -5, 17, -11, 10, 0, 9, -17, -11, -12, -10, -15, -19, -15, -10, -4, -7, 14, -6, 3, -16, -9, -6, 7, -5, -1, -3, 18, -1, -13, 7, -5, -1, 12, -3, -17, -1, -5, -10}
, {9, -12, -9, -11, 9, -13, -12, 10, 11, 7, 6, -15, -7, -17, 4, -12, 10, 13, -1, 0, 7, -16, -1, 3, -4, 6, 0, 3, 16, 12, -1, 9, -15, 7, 8, 14, 1, -4, 7, -14, 4, -12, 7, 12, 4, -16, 3, 2, 0, -13, 0, 14, 6, 12, -9, -2, 1, -2, 8, -16, 5, -16, -3, -5}
}
, {{4, 8, -15, 5, 8, 3, -3, 7, -7, -7, -5, -9, -12, -7, 9, 4, -1, -18, -3, 7, -18, -9, -3, -10, -18, 15, 15, 11, 5, -10, 8, -6, -5, -11, 15, 13, -15, -15, 3, -12, 12, -7, 1, 5, -6, 10, -1, -12, -4, -6, 12, -15, 1, -10, -4, 11, 7, -12, -10, 0, -11, -4, -14, 7}
, {12, -8, -2, -13, -4, -1, -3, 0, -7, 18, -9, -15, -16, 5, -10, -13, -7, 1, -2, 2, -15, -7, -12, -5, 8, -13, -3, 16, -6, -14, -14, -2, 7, -15, 5, 9, 0, -13, 0, -17, 9, -4, 4, -13, 17, 11, 9, -3, -9, 6, -5, 12, -8, 4, 7, -14, 4, -10, 8, -5, 9, -10, -17, 11}
, {-10, 10, 1, -5, -10, 8, -9, 11, 8, 17, -13, -2, -13, -13, 1, 3, -10, -7, 0, 14, 0, -14, 3, 10, 2, -4, 5, -1, 8, 14, -13, 3, 2, -8, -15, 6, -1, -4, -3, -5, -11, -9, 11, 8, 16, 2, 0, -15, 11, 14, 12, 11, 4, -1, 14, -6, 7, 14, -7, -14, -11, 11, -3, 2}
}
, {{10, -9, -1, 7, 1, -10, 0, -6, -5, -16, 12, -15, 4, 6, 14, -2, -7, -13, -2, -11, 16, 9, -2, -16, 8, 10, -4, 6, -5, -18, 10, -7, -17, 4, -4, -1, -3, -11, 9, 7, -14, -5, 0, -6, 3, 5, -16, -14, -13, -18, 5, 5, 15, 17, 6, -12, -3, 2, -2, -12, -13, 8, -1, 10}
, {3, 2, 11, 9, 0, -9, -5, 1, -11, -10, 11, -9, -9, -8, 9, 10, 4, -6, -7, -16, 9, -14, 11, 8, 6, -13, 7, -16, -4, 12, -5, -17, 1, 13, 9, -15, 4, 8, 13, -9, 1, -11, -1, -6, 0, 2, 7, 8, -11, 11, 6, 0, 13, 9, 4, -18, 1, -12, -14, 10, -6, -10, 14, 14}
, {5, -12, -9, -10, 0, -7, -10, 10, -1, 2, 0, -3, 8, -1, 1, 6, -16, 16, 4, 1, 3, -1, -9, -4, -7, 12, 8, 4, 7, -4, -6, 13, -16, -6, -14, 6, -3, -6, 8, 1, 0, -11, -15, -14, 12, 0, -7, 6, 5, -9, 12, 4, -7, 11, 5, 10, 7, -15, 13, 11, -12, 13, -17, -16}
}
, {{-8, -11, 8, -2, -5, 13, -9, -13, -1, -1, -13, 6, 15, 6, 12, -14, 14, -13, -7, 14, 0, -3, -3, -11, 16, 7, -3, -9, 6, 9, 2, 10, -19, -6, 3, -7, -6, 14, -15, -4, -10, -6, 10, -5, 14, 6, 9, 20, -7, -3, 5, 2, 8, 5, 9, 2, -3, 9, 10, -16, -18, 14, 12, 0}
, {-2, 6, 1, -12, 5, -13, 0, 8, 5, 8, 9, -12, -5, 4, -8, 4, -9, 19, 11, -9, 8, -14, 6, 11, -1, 3, 1, -18, 7, 4, -4, 10, -9, -4, 12, 0, -6, 7, -1, -14, -6, 7, -3, -12, 6, 14, 5, 19, 3, 0, -10, -13, 9, -4, -18, 5, -13, 6, 13, 7, 0, 2, -14, 13}
, {-11, 4, -5, -2, -9, -14, 16, -5, -7, 12, 3, -4, -11, -3, -12, -1, -6, -6, 9, 14, 13, 0, 12, -17, 4, -11, -5, 6, -13, -1, -1, -2, 7, 10, 4, 11, 13, 2, -13, -3, 15, 8, -3, 2, -11, 7, 12, -7, -11, 6, -13, -3, 10, -2, 6, -8, 6, 10, -11, -12, 3, -2, -1, 4}
}
, {{-16, -8, -8, 19, -2, -7, -4, 13, 9, -1, 13, -11, -6, 7, 12, 8, 10, 1, -8, -13, 3, 2, 10, 6, -7, -8, -8, -6, -16, 9, 5, 8, 11, 10, -13, -17, -7, -12, 6, 3, -1, -10, -12, -15, -2, 2, -10, 10, 6, -10, -11, -13, 3, -14, 2, 9, -12, 7, -10, 3, -13, -9, 7, -3}
, {4, -15, -4, 12, 7, 0, 7, 0, 7, 9, -10, 13, 10, 3, 11, -6, 12, -8, 9, 7, 2, -2, -13, 0, -2, -5, 9, -20, 5, 14, -4, -1, 7, -2, -14, 0, 3, -3, 3, -15, 11, -10, -15, 1, 7, 7, 7, 5, 14, 5, 4, 12, 14, -8, -3, 9, 13, 2, -7, 0, -8, 5, -4, -7}
, {1, 0, -12, 13, 2, -15, 7, -9, 13, -1, -15, 1, -11, -6, 9, -15, 2, 15, -16, 9, -9, -7, 11, 13, -14, -2, 11, -2, -1, 8, -16, 4, 4, 5, -8, -14, 6, 2, 9, 8, -16, -10, -1, -3, -15, 1, -4, 19, 3, 6, -7, 0, -14, -13, -1, -10, 5, 0, 0, 0, 14, -6, -5, 12}
}
, {{-10, -8, 18, -3, -4, -18, -8, 13, -11, -4, -10, -3, -10, -12, -16, 5, 4, -15, 8, -10, -14, -10, -4, -2, 1, 0, -17, -18, -15, 1, 19, 5, -12, -2, 15, 5, -8, 16, 12, 15, 16, 3, 10, -2, 7, 13, 2, 15, 10, 11, 5, -4, 7, 8, -2, -7, -17, 17, -12, -1, -3, 5, -15, -4}
, {4, 7, 1, 10, -13, -5, -9, -14, 18, 14, -8, -1, 11, -13, -11, -14, 4, -2, -4, 12, 1, -15, -1, -13, 7, 3, 6, 7, -4, 15, -6, -14, -5, 10, -14, 3, -1, 3, -8, 3, -11, 4, 14, -12, 13, 4, -1, -8, 10, -11, -6, -18, 1, 3, -12, 0, 15, 10, -3, 13, 0, -15, 13, 9}
, {-1, -11, -4, 9, 1, -17, 12, 11, 14, -13, -13, -11, 3, -16, -6, 4, 4, 4, -16, -10, 13, -3, -9, -15, 10, -14, 10, -5, 2, 12, 8, 4, 1, -3, 3, 10, 15, -9, -2, -2, 1, 1, -14, -13, 5, 17, 12, 12, 5, 4, -10, 1, 9, -13, -11, -7, 10, 6, -2, 14, 1, 9, -12, -8}
}
, {{14, -7, 13, 7, -11, -9, -15, -3, -15, -3, 15, 14, -9, 4, 4, 8, 8, -4, 10, -9, 10, 6, 1, -1, -12, 11, 0, -4, -11, 3, 10, -10, -11, 9, -1, -8, -16, -11, -2, -10, -7, -12, -5, -2, 8, -15, -6, -3, 1, 8, 12, 18, 5, -4, -7, -6, 4, 8, 12, 9, -3, -12, -12, -5}
, {-7, 4, -1, 5, 9, -17, -11, -13, 8, 7, -12, -2, -10, -8, 13, 2, -4, 6, 4, 14, -4, -4, -11, 3, 0, 5, -5, 1, -10, -16, -11, -8, -13, 8, -11, -7, 6, -7, 9, 11, -7, 12, -15, 3, 13, -6, 12, 0, 11, -4, -10, 13, -12, 4, -10, -16, 15, 2, -9, -11, -6, -10, 1, 10}
, {-10, 14, -1, -12, 8, 6, 2, 14, -10, 10, 9, -5, -8, -2, 7, -10, 8, 8, -8, -9, -11, 14, -4, 7, -1, 0, 10, 15, -2, -9, -15, 3, -2, 11, -11, 11, -13, -14, -8, -14, 13, 3, 13, -15, 7, -17, 7, -9, 5, -9, -12, 13, -3, 8, 15, 1, -8, -11, 1, 13, -1, -12, 7, -13}
}
, {{13, 13, 8, -6, -8, -3, 7, -2, -1, 12, -14, 5, -8, -9, 12, -1, 7, 7, -6, -13, -8, -16, -14, 4, 6, 8, -7, -1, -3, -9, -7, 1, 2, -1, -4, 11, 1, 1, 17, -10, 5, -5, 2, -12, -14, 6, 1, 17, 10, 4, -14, 0, 18, -4, -13, 3, 3, -6, 1, 11, 2, 9, -7, 6}
, {13, -8, -15, 6, -2, -8, -7, -16, -17, -15, -11, -9, -10, 3, 16, -16, 19, -3, 7, 1, -11, 2, 10, -12, -13, -6, -6, -2, -5, 6, -6, -13, -1, -7, -8, -9, 3, 11, -6, -11, 8, 5, -5, 2, -8, -6, 5, 2, -5, 11, 6, 6, 5, -3, -3, -1, -10, 0, -2, -14, -9, 11, -1, -4}
, {-3, -17, -16, 1, -4, -13, 2, -4, 8, -3, -15, 11, -13, 10, 7, -5, -4, 15, 11, 5, -6, -14, 13, -5, 9, -10, 12, 5, 5, 3, 1, -12, -12, -2, -9, -3, 2, 11, -3, -13, 12, -4, 13, 0, 7, 8, -14, -8, -10, 10, -12, 1, -4, 1, -6, -1, 12, -4, 3, -12, 12, -16, 13, -12}
}
, {{0, 5, -3, -6, 9, 9, 14, 0, 8, -5, 11, -10, -1, -2, -6, 8, 14, 0, 10, -5, -10, -10, -2, -16, -4, 12, -8, -15, 9, 13, -14, -15, 13, -2, 4, 7, -3, 1, -5, -1, -3, -14, -9, -13, -7, 16, -9, 11, -15, -15, 5, -16, -16, 6, -10, 0, 9, -11, -14, -11, -17, 14, -1, 4}
, {13, -13, 9, -1, 1, 4, 8, -12, -11, 3, -8, -11, -3, 8, -7, 2, 0, 3, -4, 8, 13, -2, -10, -7, -7, 4, -1, -18, -14, -13, -11, 0, -4, -2, 13, -2, 3, -5, 14, 6, -4, 15, 14, -6, 9, -7, 2, -1, 10, 8, 2, -10, 9, -7, -1, -7, 4, 20, 3, -13, 0, 2, -16, -1}
, {2, -7, 8, 6, -6, -9, 9, 9, 8, -11, 10, 14, 11, -7, 0, -8, -6, -7, -8, -8, 3, -7, 2, -9, 17, 3, 2, -8, -3, -9, 13, 1, 6, 2, -9, -9, 6, -4, 0, 7, -10, 8, 1, -4, -9, 8, 9, 5, -12, 15, -22, 13, 3, -12, -7, 9, -16, 13, 12, -16, -8, 6, -4, 12}
}
, {{18, 11, -8, 9, -7, 5, -6, 6, 8, 7, -8, -5, -3, 1, 7, -3, 13, 20, 5, 15, -15, -11, 4, -1, -7, 7, 0, -3, 15, -13, 5, 6, -13, 4, -9, -5, -2, -15, 2, -12, 14, 6, 1, -6, 13, 15, 2, 14, 14, -12, 4, -8, 16, -7, 3, -2, -14, -10, 5, -12, 13, 10, -15, 12}
, {2, 1, 0, 1, -13, 4, 13, -5, 5, -12, 10, -15, -18, -3, -3, -11, -3, 19, 12, 4, -13, -4, -9, 14, 0, -4, -6, 9, 5, 6, -9, -17, 1, -1, 6, -1, 3, -17, 11, 7, 2, 13, -2, 10, 6, 16, -13, 2, 13, -6, -4, 7, 19, 13, 5, 14, -3, -4, -2, -12, -8, 1, -3, -16}
, {-3, 7, 4, 4, -12, 4, -3, -9, -2, 5, 13, 13, 13, -13, -9, -15, 5, -2, 6, -9, -7, -7, 3, -11, 15, 3, 10, 7, -14, 16, -6, -5, 5, -4, 11, -10, -9, -2, 6, -19, -18, -1, 9, -7, 7, 1, -7, 14, -14, -9, 13, 13, 19, -5, -4, 10, -3, 6, -5, 14, -9, -16, 4, 5}
}
, {{10, 17, 15, 11, 12, 8, 9, -4, -8, 0, -17, 7, -8, -9, 15, -2, -8, -3, -1, 5, 3, 2, -4, -11, 8, 7, -8, -4, -9, -13, 4, -5, 12, -3, -6, -8, -8, 7, -16, -4, -14, -12, -12, 4, -6, 1, -16, -13, -5, -4, 2, 6, -10, 1, -4, 3, 3, -5, 14, 2, 8, 7, 1, -5}
, {-4, -8, 4, 7, -17, -8, -9, -9, 11, 3, 0, 14, 14, -7, -3, 11, -14, -14, 0, -15, -1, -8, -17, -3, -14, -14, 20, 3, 8, 13, -7, 3, 3, -6, 4, 2, 12, -8, 1, 4, 15, 6, -10, 0, -10, -4, -18, 7, 12, 11, -7, -17, -3, 10, 9, 15, 13, -7, -17, -11, -12, 1, 14, -11}
, {-8, -1, 5, -3, -11, -6, -14, -12, -5, 13, 2, -16, -5, 7, -8, 6, 10, -8, 0, 6, -13, -10, -11, 1, -6, 4, 2, -6, 13, -2, 0, 14, -2, -15, -2, 14, 8, -6, 16, 10, -13, -11, 2, -7, -8, 10, -18, -13, -3, -3, 4, 8, 1, 4, -8, -9, 13, 10, 13, 5, 7, -4, -12, -14}
}
, {{-15, 4, -3, 4, 5, -16, 0, -5, -4, -21, 6, -12, 8, 10, -5, 7, 15, 14, -1, 10, 9, -10, -4, 8, 14, -8, -15, -15, 9, -1, 11, -4, -8, 6, 12, -17, -4, 5, -2, 2, -1, 14, 0, -2, 4, -1, -17, -2, 4, 12, 16, -9, -15, -2, 10, -18, 4, 11, 6, 8, 1, -4, 4, 7}
, {-18, -8, -3, -5, 2, -17, -5, 4, 14, 3, -9, 1, 14, 5, -11, -2, 8, -12, 0, -10, 2, 13, 1, 0, 7, -15, -16, 7, 6, 7, -1, -11, -3, -15, 8, 8, -10, -12, -6, 11, -18, 5, 2, 0, -7, -9, -9, -8, -10, -1, 13, 0, 5, 0, 1, 13, 12, -8, -6, 1, -18, -16, -1, -2}
, {9, -11, -10, 7, 7, 9, -17, 5, 16, -7, 13, -8, -4, -9, -7, 3, 5, 8, -7, -1, -3, -10, 3, -15, -1, -12, -6, -5, -4, 6, 9, -2, -17, 13, 1, -11, 11, -10, 0, -14, -6, -2, -8, 9, 10, 0, 2, -12, 4, 9, -9, 4, 15, -5, 5, 0, -6, -13, -15, 8, -15, 8, -5, -3}
}
, {{8, -8, 11, 9, -11, -7, 4, -18, 11, -1, -12, -14, -4, -15, -6, -10, 13, -2, -6, -3, -14, 13, -7, 2, -4, -15, -7, 5, -10, 3, 11, -12, 0, -16, -18, -2, 16, -13, 12, -8, -15, 0, -7, -12, 13, -1, -12, 11, 10, -9, -2, -18, 9, 13, -9, 0, -8, 3, -9, 0, -11, 12, -6, -10}
, {-17, -8, 1, 2, 10, 11, -7, 8, 5, 4, 2, 4, 3, -13, 5, 7, 15, 12, 10, -1, 9, 7, -3, -7, -9, -3, -12, -5, -12, 2, 2, 1, -7, -8, 6, 1, 14, 0, -1, 5, -4, -9, 8, 3, -16, 15, -1, 11, 10, 13, -13, 8, 15, 14, 13, 13, -8, 0, -4, 6, 8, -3, 14, -14}
, {-2, 0, -10, -11, -11, -11, 3, -4, -11, -7, -12, 13, 4, -3, 14, -4, 15, -14, -16, -8, 9, -14, -8, 9, 11, -13, 1, -3, 7, 3, 3, 9, 1, 2, 1, 7, 12, -5, -11, 0, -15, 14, -10, 7, -9, 17, 0, 1, -15, 4, -7, -12, 12, -1, -17, 0, 4, -3, 13, 14, -12, -7, 6, -10}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_275_H_
#define _BATCH_NORMALIZATION_275_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       45

typedef int16_t batch_normalization_275_output_type[45][64];

#if 0
void batch_normalization_275(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_275_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_275_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_275.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       45
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_275(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_275_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_275_bias[64] = {3, 40, 7, 53, 19, 11, 3, -4, 38, 18, -6, -40, -53, 6, 19, -41, -46, -7, -10, -42, 21, 62, 46, 45, 43, -26, 35, 57, 55, 23, -20, 38, 32, 62, 12, -79, 48, -52, 62, -45, 36, 3, 20, -14, -12, -15, 24, 17, 34, 27, 26, 25, 51, 21, -51, -1, -2, 41, 37, 24, -55, 46, 27, 8}
;
const int16_t batch_normalization_275_kernel[64] = {87, 101, 129, 113, 120, 146, 209, 133, 119, 180, 138, 158, 147, 130, 109, 189, 155, 157, 162, 142, 149, 166, 124, 69, 177, 160, 163, 143, 116, 143, 171, 104, 130, 146, 132, 141, 107, 194, 116, 166, 181, 113, 134, 185, 198, 170, 92, 140, 138, 164, 108, 127, 184, 121, 193, 184, 107, 192, 122, 142, 150, 199, 200, 144}
;
/**
  ******************************************************************************
  * @file    averagepool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _AVERAGE_POOLING1D_32_H_
#define _AVERAGE_POOLING1D_32_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   45
#define POOL_SIZE       20
#define POOL_STRIDE     20
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t average_pooling1d_32_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void average_pooling1d_32(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_AVERAGE_POOLING1D_32_H_
/**
  ******************************************************************************
  * @file    averagepool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "average_pooling1d_32.h"
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   45
#define POOL_SIZE       20
#define POOL_STRIDE     20
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void average_pooling1d_32(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  LONG_NUMBER_T avg, tmp;

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
      tmp = 0;
      for (x = 0; x < POOL_SIZE; x++) {
        tmp += input[(pos_x*POOL_STRIDE)+x][k];
      }
#ifdef ACTIVATION_RELU
      if (tmp < 0) {
        tmp = 0;
      }
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation function"
#endif
      avg = tmp / POOL_SIZE;

      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, avg, INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_32_H_
#define _FLATTEN_32_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 128

typedef int16_t flatten_32_output_type[OUTPUT_DIM];

#if 0
void flatten_32(
  const number_t input[2][64], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_32_H_
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0.0
  * @date    26 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "flatten_32.h"
#include "number.h"
#endif

#define OUTPUT_DIM 128

#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t

static inline void flatten_32(
  const NUMBER_T input[2][64], 			      // IN
	NUMBER_T output[OUTPUT_DIM]) {			                // OUT

  NUMBER_T *input_flat = (NUMBER_T *)input;

  // Copy data from input to output only if input and output don't point to the same memory address already
  if (input_flat != output) {
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
      output[i] = input_flat[i];
    }
  }
}

#undef OUTPUT_DIM
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_32_H_
#define _DENSE_32_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 128
#define FC_UNITS 5

typedef int16_t dense_32_output_type[FC_UNITS];

#if 0
void dense_32(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_32_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_32.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 128
#define FC_UNITS 5
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void dense_32(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 128
#define FC_UNITS 5


const int16_t dense_32_bias[FC_UNITS] = {0, -5, -5, 5, -4}
;

const int16_t dense_32_kernel[FC_UNITS][INPUT_SAMPLES] = {{-6, 1, -6, 3, 5, -17, 3, -4, 6, -8, 9, -17, 6, 1, 12, -9, -11, -7, -11, -4, 5, 22, -10, 18, -19, -15, -22, 2, 6, 16, -6, -12, 17, -21, -8, 15, -3, 15, 8, -12, -2, 23, -2, -12, -25, -7, 14, 1, 11, 10, 8, 22, -11, 8, 10, 7, -16, -14, 9, -4, 6, 12, 7, -11, 22, -7, 14, -13, 17, -1, -12, 18, 20, -8, -20, -10, 6, -23, 15, -14, -15, 18, -5, -5, -17, 10, 22, -24, 17, 13, -4, 13, 20, -7, 8, 25, -9, -17, -16, -5, 13, 8, 3, -15, 12, 5, -6, 14, -5, -16, -23, -12, 4, 20, 12, 20, -14, 9, 20, 5, 2, -15, 6, -7, 12, 4, 13, 26}
, {19, -2, -12, 23, -4, -16, -12, 22, 6, -26, 13, -10, 18, 14, -13, -27, 17, -22, 6, 5, 12, 2, 18, 8, -13, -9, 7, 2, 1, 17, -32, -21, 10, 6, 10, 18, 7, -27, -10, -13, -11, 19, -18, -28, 8, 5, 0, 5, 4, -20, 22, 24, -10, 15, -2, 0, -1, 6, -4, 1, 0, -9, -7, -23, 23, -7, 8, -19, -14, -11, -23, -22, 17, -27, 1, 19, 18, -13, 11, -30, -10, -29, -28, -26, 7, 17, 18, -19, -17, -8, 3, 3, 2, -19, 5, 23, -8, -11, 1, 7, -18, -20, -13, 14, 20, -16, -9, -21, -5, 8, -27, 23, -5, -4, -15, 20, 15, 9, 15, -2, -14, 16, 4, 8, 7, 20, 10, -11}
, {-17, -4, 2, 10, 19, -27, 1, -18, -25, 21, 13, 23, -4, -23, 22, 18, 17, 6, 25, -8, 2, -7, -22, -17, 19, 24, 0, -28, 15, -21, 3, -5, -15, -15, 19, 6, 5, 15, 10, -8, 18, 7, -13, 12, -15, 25, -25, -18, 3, -1, -24, -5, -11, -25, -21, -7, -9, -24, -9, -16, -20, 15, 15, 6, 19, 17, -26, -6, 2, -14, 16, -15, -23, -7, 25, -7, 15, -25, 21, -15, 2, 12, -2, 15, 16, -17, 18, -10, 13, -15, 17, -22, -6, -6, 4, 21, -19, -24, -10, 0, 2, 16, 14, -21, 10, -19, 17, 1, -3, 13, -13, 16, -10, -10, 15, -14, 3, -4, 9, -20, -4, -25, -30, 8, -22, 0, -4, -16}
, {23, -8, -24, -13, -15, -3, -10, 10, -23, 23, -4, 17, 23, 8, 2, 1, 16, 24, 20, 1, -21, -2, 17, -11, 16, -3, 10, -9, -7, -11, -12, -17, 3, 6, 25, 19, 22, 14, -23, 22, 6, -22, 4, 7, 16, 23, 20, -13, 22, -25, 2, -21, -21, 5, 19, 12, 22, -13, -14, 19, -24, 12, 13, -23, -2, -2, 13, 23, 26, -6, 12, 3, -2, -12, 23, 20, 6, -1, -5, -12, 5, -15, -10, 21, -14, -3, 24, 18, 10, 12, 13, 5, -22, 19, 17, -7, -28, -18, -5, 3, 15, 19, 13, 17, -6, 11, 0, 3, -5, 8, -1, -5, 0, -14, -14, -24, -16, 7, 26, -19, -18, -2, -1, 12, -12, -7, -3, -24}
, {-18, -23, 18, 23, -24, 15, -16, 12, 15, 16, 15, -3, -19, -23, -3, -22, -29, 13, -3, -24, 14, -5, -15, -27, 15, -27, 8, 7, 0, 2, 13, 19, 16, 8, -5, -23, 6, -8, -7, 18, -5, -20, 5, 11, -25, 7, -8, -11, -18, 21, -11, -2, -23, -14, 24, 25, -1, -24, 20, 24, 19, -19, 4, -7, 6, 19, 25, 6, -6, 14, 21, 17, 3, 1, -12, -21, -4, -5, 0, 14, 16, 2, 11, -20, 1, 17, 0, -23, 0, -27, -27, 2, -24, -8, -14, -10, -12, -1, -18, -14, -6, -5, 12, 20, -2, 11, -17, -4, -12, 8, -3, -11, -11, 16, -23, -12, -17, 20, 4, 19, -12, 1, -16, 16, -13, -18, 24, 17}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"

 // InputLayer is excluded
#include "conv1d_337.h" // InputLayer is excluded
#include "batch_normalization_262.h" // InputLayer is excluded
#include "max_pooling1d_92.h" // InputLayer is excluded
#include "conv1d_338.h" // InputLayer is excluded
#include "batch_normalization_263.h" // InputLayer is excluded
#include "conv1d_339.h" // InputLayer is excluded
#include "batch_normalization_264.h" // InputLayer is excluded
#include "conv1d_340.h" // InputLayer is excluded
#include "batch_normalization_265.h" // InputLayer is excluded
#include "conv1d_341.h" // InputLayer is excluded
#include "batch_normalization_266.h" // InputLayer is excluded
#include "conv1d_342.h" // InputLayer is excluded
#include "batch_normalization_267.h" // InputLayer is excluded
#include "conv1d_343.h" // InputLayer is excluded
#include "batch_normalization_268.h" // InputLayer is excluded
#include "conv1d_344.h" // InputLayer is excluded
#include "batch_normalization_269.h" // InputLayer is excluded
#include "conv1d_345.h" // InputLayer is excluded
#include "batch_normalization_270.h" // InputLayer is excluded
#include "conv1d_346.h" // InputLayer is excluded
#include "batch_normalization_271.h" // InputLayer is excluded
#include "conv1d_347.h" // InputLayer is excluded
#include "batch_normalization_272.h" // InputLayer is excluded
#include "conv1d_348.h" // InputLayer is excluded
#include "batch_normalization_273.h" // InputLayer is excluded
#include "conv1d_349.h" // InputLayer is excluded
#include "batch_normalization_274.h" // InputLayer is excluded
#include "conv1d_350.h" // InputLayer is excluded
#include "batch_normalization_275.h" // InputLayer is excluded
#include "average_pooling1d_32.h" // InputLayer is excluded
#include "flatten_32.h" // InputLayer is excluded
#include "dense_32.h"
#endif


#define MODEL_INPUT_DIM_0 500
#define MODEL_INPUT_DIM_1 1
#define MODEL_INPUT_DIMS 500 * 1

#define MODEL_OUTPUT_SAMPLES 5

#define MODEL_INPUT_SCALE_FACTOR 7 // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_FLOOR
#define MODEL_INPUT_NUMBER_T int16_t
#define MODEL_INPUT_LONG_NUMBER_T int32_t

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[500][1];
typedef int16_t input_t[500][1];
typedef dense_32_output_type output_t;


void cnn(
  const input_t input,
  output_t output);

void reset(void);

#endif//__MODEL_H__


#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"
// #include <chrono>

 // InputLayer is excluded
#include "conv1d_337.c"
#include "weights/conv1d_337.c" // InputLayer is excluded
#include "batch_normalization_262.c"
#include "weights/batch_normalization_262.c" // InputLayer is excluded
#include "max_pooling1d_92.c" // InputLayer is excluded
#include "conv1d_338.c"
#include "weights/conv1d_338.c" // InputLayer is excluded
#include "batch_normalization_263.c"
#include "weights/batch_normalization_263.c" // InputLayer is excluded
#include "conv1d_339.c"
#include "weights/conv1d_339.c" // InputLayer is excluded
#include "batch_normalization_264.c"
#include "weights/batch_normalization_264.c" // InputLayer is excluded
#include "conv1d_340.c"
#include "weights/conv1d_340.c" // InputLayer is excluded
#include "batch_normalization_265.c"
#include "weights/batch_normalization_265.c" // InputLayer is excluded
#include "conv1d_341.c"
#include "weights/conv1d_341.c" // InputLayer is excluded
#include "batch_normalization_266.c"
#include "weights/batch_normalization_266.c" // InputLayer is excluded
#include "conv1d_342.c"
#include "weights/conv1d_342.c" // InputLayer is excluded
#include "batch_normalization_267.c"
#include "weights/batch_normalization_267.c" // InputLayer is excluded
#include "conv1d_343.c"
#include "weights/conv1d_343.c" // InputLayer is excluded
#include "batch_normalization_268.c"
#include "weights/batch_normalization_268.c" // InputLayer is excluded
#include "conv1d_344.c"
#include "weights/conv1d_344.c" // InputLayer is excluded
#include "batch_normalization_269.c"
#include "weights/batch_normalization_269.c" // InputLayer is excluded
#include "conv1d_345.c"
#include "weights/conv1d_345.c" // InputLayer is excluded
#include "batch_normalization_270.c"
#include "weights/batch_normalization_270.c" // InputLayer is excluded
#include "conv1d_346.c"
#include "weights/conv1d_346.c" // InputLayer is excluded
#include "batch_normalization_271.c"
#include "weights/batch_normalization_271.c" // InputLayer is excluded
#include "conv1d_347.c"
#include "weights/conv1d_347.c" // InputLayer is excluded
#include "batch_normalization_272.c"
#include "weights/batch_normalization_272.c" // InputLayer is excluded
#include "conv1d_348.c"
#include "weights/conv1d_348.c" // InputLayer is excluded
#include "batch_normalization_273.c"
#include "weights/batch_normalization_273.c" // InputLayer is excluded
#include "conv1d_349.c"
#include "weights/conv1d_349.c" // InputLayer is excluded
#include "batch_normalization_274.c"
#include "weights/batch_normalization_274.c" // InputLayer is excluded
#include "conv1d_350.c"
#include "weights/conv1d_350.c" // InputLayer is excluded
#include "batch_normalization_275.c"
#include "weights/batch_normalization_275.c" // InputLayer is excluded
#include "average_pooling1d_32.c" // InputLayer is excluded
#include "flatten_32.c" // InputLayer is excluded
#include "dense_32.c"
#include "weights/dense_32.c"
#endif


void cnn(
  const input_t input,
  dense_32_output_type dense_32_output) {
  
  // Output array allocation
  static union {
    conv1d_337_output_type conv1d_337_output;
    max_pooling1d_92_output_type max_pooling1d_92_output;
    batch_normalization_263_output_type batch_normalization_263_output;
    batch_normalization_264_output_type batch_normalization_264_output;
    batch_normalization_265_output_type batch_normalization_265_output;
    batch_normalization_266_output_type batch_normalization_266_output;
    batch_normalization_267_output_type batch_normalization_267_output;
    batch_normalization_268_output_type batch_normalization_268_output;
    batch_normalization_269_output_type batch_normalization_269_output;
    batch_normalization_270_output_type batch_normalization_270_output;
    batch_normalization_271_output_type batch_normalization_271_output;
    batch_normalization_272_output_type batch_normalization_272_output;
    batch_normalization_273_output_type batch_normalization_273_output;
    batch_normalization_274_output_type batch_normalization_274_output;
    batch_normalization_275_output_type batch_normalization_275_output;
  } activations1;

  static union {
    batch_normalization_262_output_type batch_normalization_262_output;
    conv1d_338_output_type conv1d_338_output;
    conv1d_339_output_type conv1d_339_output;
    conv1d_340_output_type conv1d_340_output;
    conv1d_341_output_type conv1d_341_output;
    conv1d_342_output_type conv1d_342_output;
    conv1d_343_output_type conv1d_343_output;
    conv1d_344_output_type conv1d_344_output;
    conv1d_345_output_type conv1d_345_output;
    conv1d_346_output_type conv1d_346_output;
    conv1d_347_output_type conv1d_347_output;
    conv1d_348_output_type conv1d_348_output;
    conv1d_349_output_type conv1d_349_output;
    conv1d_350_output_type conv1d_350_output;
    average_pooling1d_32_output_type average_pooling1d_32_output;
    flatten_32_output_type flatten_32_output;
  } activations2;


// Model layers call chain 
  
  
  conv1d_337( // First layer uses input passed as model parameter
    input,
    conv1d_337_kernel,
    conv1d_337_bias,
    activations1.conv1d_337_output
    );
  
  
  batch_normalization_262(
    activations1.conv1d_337_output,
    batch_normalization_262_kernel,
    batch_normalization_262_bias,
    activations2.batch_normalization_262_output
    );
  
  
  max_pooling1d_92(
    activations2.batch_normalization_262_output,
    activations1.max_pooling1d_92_output
    );
  
  
  conv1d_338(
    activations1.max_pooling1d_92_output,
    conv1d_338_kernel,
    conv1d_338_bias,
    activations2.conv1d_338_output
    );
  
  
  batch_normalization_263(
    activations2.conv1d_338_output,
    batch_normalization_263_kernel,
    batch_normalization_263_bias,
    activations1.batch_normalization_263_output
    );
  
  
  conv1d_339(
    activations1.batch_normalization_263_output,
    conv1d_339_kernel,
    conv1d_339_bias,
    activations2.conv1d_339_output
    );
  
  
  batch_normalization_264(
    activations2.conv1d_339_output,
    batch_normalization_264_kernel,
    batch_normalization_264_bias,
    activations1.batch_normalization_264_output
    );
  
  
  conv1d_340(
    activations1.batch_normalization_264_output,
    conv1d_340_kernel,
    conv1d_340_bias,
    activations2.conv1d_340_output
    );
  
  
  batch_normalization_265(
    activations2.conv1d_340_output,
    batch_normalization_265_kernel,
    batch_normalization_265_bias,
    activations1.batch_normalization_265_output
    );
  
  
  conv1d_341(
    activations1.batch_normalization_265_output,
    conv1d_341_kernel,
    conv1d_341_bias,
    activations2.conv1d_341_output
    );
  
  
  batch_normalization_266(
    activations2.conv1d_341_output,
    batch_normalization_266_kernel,
    batch_normalization_266_bias,
    activations1.batch_normalization_266_output
    );
  
  
  conv1d_342(
    activations1.batch_normalization_266_output,
    conv1d_342_kernel,
    conv1d_342_bias,
    activations2.conv1d_342_output
    );
  
  
  batch_normalization_267(
    activations2.conv1d_342_output,
    batch_normalization_267_kernel,
    batch_normalization_267_bias,
    activations1.batch_normalization_267_output
    );
  
  
  conv1d_343(
    activations1.batch_normalization_267_output,
    conv1d_343_kernel,
    conv1d_343_bias,
    activations2.conv1d_343_output
    );
  
  
  batch_normalization_268(
    activations2.conv1d_343_output,
    batch_normalization_268_kernel,
    batch_normalization_268_bias,
    activations1.batch_normalization_268_output
    );
  
  
  conv1d_344(
    activations1.batch_normalization_268_output,
    conv1d_344_kernel,
    conv1d_344_bias,
    activations2.conv1d_344_output
    );
  
  
  batch_normalization_269(
    activations2.conv1d_344_output,
    batch_normalization_269_kernel,
    batch_normalization_269_bias,
    activations1.batch_normalization_269_output
    );
  
  
  conv1d_345(
    activations1.batch_normalization_269_output,
    conv1d_345_kernel,
    conv1d_345_bias,
    activations2.conv1d_345_output
    );
  
  
  batch_normalization_270(
    activations2.conv1d_345_output,
    batch_normalization_270_kernel,
    batch_normalization_270_bias,
    activations1.batch_normalization_270_output
    );
  
  
  conv1d_346(
    activations1.batch_normalization_270_output,
    conv1d_346_kernel,
    conv1d_346_bias,
    activations2.conv1d_346_output
    );
  
  
  batch_normalization_271(
    activations2.conv1d_346_output,
    batch_normalization_271_kernel,
    batch_normalization_271_bias,
    activations1.batch_normalization_271_output
    );
  
  
  conv1d_347(
    activations1.batch_normalization_271_output,
    conv1d_347_kernel,
    conv1d_347_bias,
    activations2.conv1d_347_output
    );
  
  
  batch_normalization_272(
    activations2.conv1d_347_output,
    batch_normalization_272_kernel,
    batch_normalization_272_bias,
    activations1.batch_normalization_272_output
    );
  
  
  conv1d_348(
    activations1.batch_normalization_272_output,
    conv1d_348_kernel,
    conv1d_348_bias,
    activations2.conv1d_348_output
    );
  
  
  batch_normalization_273(
    activations2.conv1d_348_output,
    batch_normalization_273_kernel,
    batch_normalization_273_bias,
    activations1.batch_normalization_273_output
    );
  
  
  conv1d_349(
    activations1.batch_normalization_273_output,
    conv1d_349_kernel,
    conv1d_349_bias,
    activations2.conv1d_349_output
    );
  
  
  batch_normalization_274(
    activations2.conv1d_349_output,
    batch_normalization_274_kernel,
    batch_normalization_274_bias,
    activations1.batch_normalization_274_output
    );
  
  
  conv1d_350(
    activations1.batch_normalization_274_output,
    conv1d_350_kernel,
    conv1d_350_bias,
    activations2.conv1d_350_output
    );
  
  
  batch_normalization_275(
    activations2.conv1d_350_output,
    batch_normalization_275_kernel,
    batch_normalization_275_bias,
    activations1.batch_normalization_275_output
    );
  
  
  average_pooling1d_32(
    activations1.batch_normalization_275_output,
    activations2.average_pooling1d_32_output
    );
  
  
  flatten_32(
    activations2.average_pooling1d_32_output,
    activations2.flatten_32_output
    );
  
  
  dense_32(
    activations2.flatten_32_output,
    dense_32_kernel,
    dense_32_bias,// Last layer uses output passed as model parameter
    dense_32_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif
