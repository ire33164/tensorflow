/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_INTERMITTENT_INTEGER_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_INTERMITTENT_INTEGER_FULLY_CONNECTED_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/simulate_nvm.h"

namespace tflite {
namespace intermittent_integer_ops {

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  printf("Intermittent FC\n");
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  // for intermitttent
  const int input_length = input_shape.Dims(1) * input_shape.Dims(2) * output_depth;
  const int output_length = output_depth;
  int tmp_batch = 0;
  int tmp_channel = 0;

  if (!is_power_failure) {
    // Backing up entire input data to NVM in order to avoid lose input data.
    intermittent_params[offset_nvm].input_version = !intermittent_params[offset_nvm].input_version;
    write_to_nvm_segmented(const_cast<int8_t *>(input_data), intermittent_params[offset_nvm].input_version ? NODE_INPUT2 : NODE_INPUT1, input_length, MAX_ACCESS_LENGTH);
  } else {
    // Recover the node's input and output in VM.
    read_from_nvm_segmented(const_cast<int8_t *>(input_data), intermittent_params[offset_nvm].input_version ? NODE_INPUT2 : NODE_INPUT1, input_length, MAX_ACCESS_LENGTH);
    read_from_nvm_segmented(reinterpret_cast<void *>(output_data), offset_nvm ? NODE_OUTPUT2 : NODE_OUTPUT1, output_length, MAX_ACCESS_LENGTH);
    int OFM_cnt = intermittent_params[offset_nvm].OFM_cnt;
    tmp_batch = OFM_cnt / output_depth;
    tmp_channel = OFM_cnt - tmp_batch * output_depth;
  }

  size_t node_idx;
  bool input_version;
  node_idx = intermittent_params[offset_nvm].node_idx;
  input_version = intermittent_params[offset_nvm].input_version;

  for (int b = 0; b < batches; ++b) {
    if (is_power_failure) b = tmp_batch;
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      if (is_power_failure) {
        version = intermittent_params[offset_nvm].version + 1;
        out_c = tmp_channel + 1;
        is_power_failure = false;
        offset_nvm = !offset_nvm;
      }

      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
      // for intermittent
      write_to_nvm_segmented(reinterpret_cast<void *>(output_data), offset_nvm ? NODE_OUTPUT2 : NODE_OUTPUT1, output_length, MAX_ACCESS_LENGTH);
      // Checkpoint forward progress infomation
      intermittent_params[offset_nvm].node_idx = node_idx;
      intermittent_params[offset_nvm].input_version = input_version;
      intermittent_params[offset_nvm].OFM_cnt = b * output_depth + out_c;
      intermittent_params[offset_nvm].version = version;
      write_to_nvm(&intermittent_params[offset_nvm], offset_nvm ? OFFSET2 : OFFSET1, sizeof(TfLiteIntermittentParams));
      ++version;
      offset_nvm = !offset_nvm;
    }
  }
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int64_t* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int64_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * input_val;
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
    }
  }
}

}  // namespace intermittent_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_INTERMITTENT_INTEGER_FULLY_CONNECTED_H_
